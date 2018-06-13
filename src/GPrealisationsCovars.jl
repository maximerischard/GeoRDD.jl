using GaussianProcesses: Mean, Kernel, num_params
using GaussianProcesses: grad_stack, grad_stack!, grad_slice!, get_ααinvcKI!
using GaussianProcesses: MatF64, cov!, addcov!, KernelData
import GaussianProcesses: get_params, set_params!, optimize!
import GaussianProcesses: update_mll_and_dmll!, update_mll!
import Base: mean

const MultiGP = Vector{GPE}

type MultiGPCovars{MT<:Mean, KT1<:Kernel, KT2<:Kernel}
    D::Array{Float64,2}
    y::Vector{Float64}
    mgp::MultiGP
    p::Int
    dim::Int
    nobsv::Int
    logNoise::Float64
    m::MT
    k::KT1
    βkern::KT2
    βdata::KernelData
    # Auxiliary data
    cK::PDMats.PDMat        # (k + obsNoise)
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    mll::Float64            # Marginal log-likelihood
    dmll::Vector{Float64}   # Gradient marginal log-likelihood
    function MultiGPCovars{MT,KT1,KT2}(D::Array{Float64,2}, 
        y::Vector{Float64},
        mgp::MultiGP, 
        p::Int,
        dim::Int,
        nobsv::Int,
        logNoise::Float64,
        m::MT,
        k::KT1,
        βkern::KT2
        ) where {MT<:Mean, KT1<:Kernel, KT2<:Kernel}
        βdata=KernelData(βkern, D')
        mgpcv = new(D, y, mgp, p, dim, nobsv, logNoise, m, k, βkern, βdata)
        initialise_mll!(mgpcv)
        return mgpcv
    end
end
function MultiGPCovars{KT2<:Kernel}(D::Array{Float64,2}, mgp::MultiGP, βkern::KT2)
    nobsv = sum([gp.nobsv for gp in mgp])
    size(D,1) == nobsv || throw(ArgumentError("incompatible dimensions of covariates matrix and gaussian processes"))
    first_gp = mgp[1]
    dim = first_gp.dim
    logNoise = first_gp.logNoise
    k = first_gp.k
    m = first_gp.m
    # harmonize parameters
    for gp in mgp
        gp.k = k
        gp.m = m
        gp.logNoise = logNoise
    end
    p = size(D,2)
    y = vcat([gp.y for gp in mgp]...)
    mgpcv = MultiGPCovars{typeof(m),typeof(k),KT2}(D, y, mgp, p, dim, nobsv, logNoise, m, k, βkern)
    return mgpcv
end
function propagate_params!(mgpcv::MultiGPCovars)
    for gp in mgpcv.mgp
        # harmonize parameters
        gp.k = mgpcv.k
        gp.m = mgpcv.m
        gp.logNoise = mgpcv.logNoise
    end
end

function update_mll!(mgpcv::MultiGPCovars)
    propagate_params!(mgpcv)
    cK = mgpcv.cK.mat
    cov!(cK, mgpcv.βkern, mgpcv.D', mgpcv.βdata)
    μ = Array{Float64}(mgpcv.nobsv)
    istart=0
    for gp in mgpcv.mgp
        μ[istart+1:istart+gp.nobsv] = mean(mgpcv.m,gp.X)
        addcov!(view(cK, istart+1:istart+gp.nobsv, istart+1:istart+gp.nobsv), mgpcv.k, gp.X, gp.data)
        istart += gp.nobsv
    end
    for i in 1:mgpcv.nobsv
        cK[i,i] += max(exp(2*mgpcv.logNoise),1e-8)
    end
    chol_buffer = mgpcv.cK.chol.factors
    copy!(chol_buffer, cK)
    chol = cholfact!(Symmetric(chol_buffer))
    mgpcv.cK = PDMats.PDMat(cK, chol)
    mgpcv.alpha = mgpcv.cK \ (mgpcv.y - μ)
    mgpcv.mll = -dot((mgpcv.y - μ),mgpcv.alpha)/2.0 - logdet(mgpcv.cK)/2.0 - mgpcv.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

function initialise_mll!(mgpcv::MultiGPCovars)
    cK = Array{Float64}(mgpcv.nobsv, mgpcv.nobsv)
    propagate_params!(mgpcv)
    cov!(cK, mgpcv.βkern, mgpcv.D', mgpcv.βdata)
    μ = Array{Float64}(mgpcv.nobsv)
    istart=0
    for gp in mgpcv.mgp
        μ[istart+1:istart+gp.nobsv] = mean(mgpcv.m,gp.X)
        addcov!(view(cK, istart+1:istart+gp.nobsv, istart+1:istart+gp.nobsv), mgpcv.k, gp.X, gp.data)
        istart += gp.nobsv
    end
    for i in 1:mgpcv.nobsv
        cK[i,i] += max(exp(2*mgpcv.logNoise),1e-8)
    end
    mgpcv.cK = PDMats.PDMat(cK)
    mgpcv.alpha = mgpcv.cK \ (mgpcv.y .- μ)
    mgpcv.mll = -dot((mgpcv.y-μ),mgpcv.alpha)/2.0 - logdet(mgpcv.cK)/2.0 - mgpcv.nobsv*log(2π)/2.0
end


function update_mll_and_dmll!(mgpcv::MultiGPCovars,
    Kgrad::MatF64,
    ααinvcKI::MatF64
    ; 
    noise::Bool=true, # include gradient component for the logNoise term
    mean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    beta::Bool=true, # include gradient components for the linear regression prior terms
    )
    update_mll!(mgpcv)
    n_mean_params = num_params(mgpcv.m)
    n_kern_params = num_params(mgpcv.k)
    n_beta_params = num_params(mgpcv.βkern)
    dmll = Array{Float64}(noise + mean*n_mean_params + kern*n_kern_params + beta*n_beta_params)
    logNoise = mgpcv.logNoise
    get_ααinvcKI!(ααinvcKI, mgpcv.cK, mgpcv.alpha)
    i=1
    if noise
        dmll[i] = exp(2*logNoise)*trace(ααinvcKI)
        i+=1
    end
    if mean
        Mgrads = vcat([grad_stack(gp.m, gp.X) for gp in mgpcv.mgp]...)
        for j in 1:n_mean_params
            dmll[i] = dot(Mgrads[:,j],α)
            i+=1
        end
    end
    if kern
        Kgrad[:,:] = 0.0
        for j in i:i+n_kern_params-1
            dmll[j] = 0.0
        end
        for iparam in 1:n_kern_params
            istart=0
            for gp in mgpcv.mgp
                Kview = view(Kgrad, istart+1:istart+gp.nobsv, istart+1:istart+gp.nobsv)
                ααview = view(ααinvcKI, istart+1:istart+gp.nobsv, istart+1:istart+gp.nobsv)
                grad_slice!(Kview, mgpcv.k, gp.X, gp.data, iparam)
                dmll[i] += vecdot(Kview, ααview)/2
                istart += gp.nobsv
            end
            i+=1
        end
    end
    if beta
        for iparam in 1:num_params(mgpcv.βkern)
            grad_slice!(Kgrad, mgpcv.βkern, mgpcv.D', mgpcv.βdata, iparam)
            dmll[i] = dot(ααinvcKI,Kgrad)/2.0
            i+=1
        end
    end
    mgpcv.dmll = dmll
end
function get_params(mgpcv::MultiGPCovars; noise::Bool=true, mean::Bool=true, kern::Bool=true, beta::Bool=true)
    params = Float64[]
    if noise; push!(params, mgpcv.logNoise); end
    if mean;  append!(params, get_params(mgpcv.m)); end
    if kern; append!(params,  get_params(mgpcv.k)); end
    if beta; append!(params,  get_params(mgpcv.βkern)); end
    return params
end
function set_params!(mgpcv::MultiGPCovars, hyp::Vector{Float64}; 
                    noise::Bool=true, mean::Bool=true, kern::Bool=true, beta::Bool=true)
    i=1
    if noise
        mgpcv.logNoise = hyp[i]
        i+=1
    end
    if mean
        set_params!(mgpcv.m, hyp[i:i+num_params(mgpcv.m)-1])
        i+=num_params(mgpcv.m)
    end
    if kern
        set_params!(mgpcv.k, hyp[i:i+num_params(mgpcv.k)-1])
        i+=num_params(mgpcv.k)
    end
    if beta
        set_params!(mgpcv.βkern, hyp[i:i+num_params(mgpcv.βkern)-1])
        i+=num_params(mgpcv.βkern)
    end
    propagate_params!(mgpcv)
end

function optimize!(mgpcv::MultiGPCovars; noise::Bool=true, mean::Bool=true, kern::Bool=true, beta::Bool=true, 
                    method=ConjugateGradient(), options=Optim.Options())
    cK_buffer = Array{Float64}(mgpcv.nobsv, mgpcv.nobsv)
    Kgrad_buffer = Array{Float64}(mgpcv.nobsv, mgpcv.nobsv)
    function mll(hyp::Vector{Float64})
        try
            set_params!(mgpcv, hyp; noise=noise, mean=mean, kern=kern, beta=beta)
            update_mll!(mgpcv)
            return -mgpcv.mll
        catch err
             if !all(isfinite(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end        
    end

    function mll_and_dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        try
            set_params!(mgpcv, hyp; noise=noise, mean=mean, kern=kern, beta=beta)
            update_mll_and_dmll!(mgpcv, cK_buffer, Kgrad_buffer; noise=noise, mean=mean, kern=kern, beta=beta)
            grad[:] = -mgpcv.dmll
            return -mgpcv.mll
        catch err
             if !all(isfinite(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end 
    end
    function dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        mll_and_dmll!(grad, hyp)
    end

    init = get_params(mgpcv;  noise=noise, mean=mean, kern=kern, beta=beta)  # Initial hyperparameter values
    func = OnceDifferentiable(mll, dmll!, mll_and_dmll!, init)
    results=optimize(func,init, method, options)
    set_params!(mgpcv, Optim.minimizer(results); noise=noise, mean=mean, kern=kern, beta=beta)
    mll(Optim.minimizer(results))
    return results
end

"""
    Obtain the full (block diagonal) spatial covariance matrix
"""
function spatial_cov(mgpcv::MultiGPCovars)
    cK = zeros(mgpcv.nobsv, mgpcv.nobsv)
    istart=0
    for gp in mgpcv.mgp
        addcov!(view(cK, istart+1:istart+gp.nobsv, istart+1:istart+gp.nobsv), mgpcv.k, gp.X, gp.data)
        istart += gp.nobsv
    end
    return cK
end

"""
    Obtain the covariance matrix of the outcome conditional
    on the β coefficients.
    cov(Y ∣ β)
"""
function get_ΣYβ(mgpcv::MultiGPCovars)
    cK = spatial_cov(mgpcv)
    for i in 1:mgpcv.nobsv
        cK[i,i] += max(exp(2*mgpcv.logNoise),1e-8)
    end
    return PDMats.PDMat(cK)
end

"""
    Evaluate the mean function.
"""
function mean(mgpcv::MultiGPCovars)
    μ = Array{Float64}(mgpcv.nobsv)
    istart=0
    for gp in mgpcv.mgp
        μ[istart+1:istart+gp.nobsv] = mean(mgpcv.m,gp.X)
        istart += gp.nobsv
    end
    return μ
end    

"""
    Get the posterior mean coefficient value E(β ∣ Y)
    Notation:
        ΣYβ = cov(Y ∣ β)
        Σβ = cov(β) = σβ² I # prior variance for β


    precision = Dᵀ ΣYβ⁻¹ D + Σβ⁻¹
    βhat = precision-weighted average
         = (precision⁻¹ D) (ΣYβ⁻¹ (Y-μ))
    
    See:
        Notes on “Analytical covariates”
        BDA3 equation 14.4
"""
function postmean_β(mgpcv::MultiGPCovars)
    ΣY_β = get_ΣYβ(mgpcv)
    precision = PDMats.X_invA_Xt(ΣY_β, mgpcv.D')
    for i in 1:mgpcv.p
        precision[i,i] += mgpcv.βkern.ℓ2
    end
    m = mean(mgpcv)
    βhat = (precision \ mgpcv.D') * (ΣY_β \ (mgpcv.y.-m))
    return βhat
end

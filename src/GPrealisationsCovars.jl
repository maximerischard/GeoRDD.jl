import Base: mean

type MultiGPCovars{KEY}
    D::Array{Float64,2}
    y::Vector{Float64}
    groupKeys::Vector{KEY}
    mgp::Dict{KEY,GPE}
    groupIndices::Dict{KEY,UnitRange{Int64}}
    p::Int
    dim::Int
    nobsv::Int
    logNoise::Float64
    m::Mean
    k::Kernel
    βkern::Kernel
    βdata::KernelData
    # Auxiliary data
    cK::PDMats.PDMat        # (k + obsNoise)
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    mll::Float64            # Marginal log-likelihood
    dmll::Vector{Float64}   # Gradient marginal log-likelihood
    function MultiGPCovars{KEY}(
            D::Array{Float64,2}, 
            y::Vector{Float64},
            groupKeys::Vector{KEY},
            mgp::Dict{KEY,GPE}, 
            groupIndices::Dict{KEY,UnitRange{Int64}},
            p::Int,
            dim::Int,
            nobsv::Int,
            logNoise::Float64,
            m::Mean,
            k::Kernel,
            βkern::Kernel
            ) where {KEY}
        size(D, 2) == nobsv || throw("incompatible dimensions of covariates matrix and gaussian processes")
        size(D, 1) == p || throw("first dimension of D is not p")
        length(groupKeys) == length(groupIndices) || throw("groupKeys and groupIndices should have same length")
        length(groupKeys) == length(mgp) || throw("groupKeys and mgp should have same length")
        βdata=KernelData(βkern, D)
        mgpcv = new(D, y, groupKeys, mgp, groupIndices, p, dim, nobsv, logNoise, m, k, βkern, βdata)
        propagate_params!(mgpcv)
        initialise_mll!(mgpcv)
        return mgpcv
    end
end
function MultiGPCovars(Dlist::Vector{M}, gpList::Vector{GPE}, groupKeys::Vector{KEY}, βkern::Kernel) where {M<:AbstractMatrix{Float64}, KEY}
    total_nobsv = sum([gp.nobsv for gp in gpList])
    D = hcat(Dlist...)
    size(D, 2) == total_nobsv || throw(ArgumentError("incompatible dimensions of covariates matrix and gaussian processes"))
    p = size(D,1)

    # get kernel and parameters from first GP in list
    first_gp = gpList[1]
    dim = first_gp.dim
    logNoise = first_gp.logNoise
    kern = first_gp.k
    m = first_gp.m
    ngroups = length(groupKeys)
    @assert ngroups == length(gpList)
    gpDict = Dict{KEY, GPE}()
    groupIndices = Dict{KEY, UnitRange{Int64}}()
    istart = 1
    for j in 1:ngroups
        gp = gpList[j]
        key = groupKeys[j]
        nobsv = gp.nobsv
        gp.nobsv > 0 || throw("empty group")
        groupIndices[key] = istart:istart+nobsv-1
        gpDict[key] = gp
        istart += nobsv
    end
    y = vcat([gp.y for gp in gpList]...)
    mgpcv = MultiGPCovars{KEY}(
                D, y, groupKeys, gpDict, groupIndices, 
                p, dim, total_nobsv, logNoise, m, kern, βkern)
    return mgpcv
end
function propagate_params!(mgpcv::MultiGPCovars)
    for gp in values(mgpcv.mgp)
        # harmonize parameters
        gp.k = mgpcv.k
        gp.m = mgpcv.m
        gp.logNoise = mgpcv.logNoise
    end
end

function update_mll!(mgpcv::MultiGPCovars)
    propagate_params!(mgpcv)
    cK = mgpcv.cK.mat
    cov!(cK, mgpcv.βkern, mgpcv.D, mgpcv.βdata)
    μ = Array{Float64}(mgpcv.nobsv)
    for key in mgpcv.groupKeys
        gp = mgpcv.mgp[key]
        slice = mgpcv.groupIndices[key]
        μ[slice] = mean(mgpcv.m,gp.X)
        addcov!(view(cK, slice, slice), mgpcv.k, gp.X, gp.data)
    end
    add_diag!(cK, max(exp(2*mgpcv.logNoise),1e-8))
    mgpcv.cK = update_chol!(mgpcv.cK)
    mgpcv.alpha = mgpcv.cK \ (mgpcv.y - μ)
    mgpcv.mll = -dot((mgpcv.y - μ),mgpcv.alpha)/2.0 - logdet(mgpcv.cK)/2.0 - mgpcv.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

function initialise_mll!(mgpcv::MultiGPCovars)
    cK = Array{Float64}(mgpcv.nobsv, mgpcv.nobsv)
    propagate_params!(mgpcv)
    cov!(cK, mgpcv.βkern, mgpcv.D, mgpcv.βdata)
    μ = Array{Float64}(mgpcv.nobsv)
    for key in mgpcv.groupKeys
        gp = mgpcv.mgp[key]
        slice = mgpcv.groupIndices[key]
        μ[slice] = mean(mgpcv.m,gp.X)
        addcov!(view(cK, slice, slice), mgpcv.k, gp.X, gp.data)
    end
    add_diag!(cK, max(exp(2*mgpcv.logNoise),1e-8))
    mgpcv.cK = PDMats.PDMat(cK)
    mgpcv.alpha = mgpcv.cK \ (mgpcv.y .- μ)
    mgpcv.mll = -dot((mgpcv.y-μ),mgpcv.alpha)/2.0 - logdet(mgpcv.cK)/2.0 - mgpcv.nobsv*log(2π)/2.0
end


function update_mll_and_dmll!(
        mgpcv::MultiGPCovars, ααinvcKI::MatF64
        ; 
        noise::Bool=true,  # include gradient component for the logNoise term
        domean::Bool=true, # include gradient components for the mean parameters
        kern::Bool=true,   # include gradient components for the spatial kernel parameters
        beta::Bool=true,   # include gradient components for the linear regression prior terms
    )
    update_mll!(mgpcv)
    n_mean_params = num_params(mgpcv.m)
    n_kern_params = num_params(mgpcv.k)
    n_beta_params = num_params(mgpcv.βkern)
    dmll = Array{Float64}(noise + domean*n_mean_params + kern*n_kern_params + beta*n_beta_params)
    logNoise = mgpcv.logNoise
    get_ααinvcKI!(ααinvcKI, mgpcv.cK, mgpcv.alpha)
    i=1
    if noise
        dmll[i] = exp(2*logNoise)*trace(ααinvcKI)
        i+=1
    end
    if domean
        Mgrads = vcat([grad_stack(gp.m, gp.X) for gp in values(mgpcv.mgp)]...)
        for j in 1:n_mean_params
            dmll[i] = dot(Mgrads[:,j],mgpcv.alpha)
            i+=1
        end
    end
    if kern
        dmll_k = @view(dmll[i:i+n_kern_params-1])
        fill!(dmll_k, 0.0)
        for key in mgpcv.groupKeys
            gp = mgpcv.mgp[key]
            slice = mgpcv.groupIndices[key]
            ααview = view(ααinvcKI, slice, slice)
            dmll_k_gp = Vector{Float64}(n_kern_params)
            dmll_kern!(dmll_k_gp, gp.k, gp.X, gp.data, ααview)
            dmll_k .+= dmll_k_gp
        end
        i += n_kern_params
    end
    if beta
        dmll_β = @view(dmll[i:end])
        dmll_kern!(dmll_β, mgpcv.βkern, mgpcv.D, mgpcv.βdata, ααinvcKI)
        i+=1
    end
    mgpcv.dmll = dmll
end
function get_params(mgpcv::MultiGPCovars; noise::Bool=true, domean::Bool=true, kern::Bool=true, beta::Bool=true)
    params = Float64[]
    if noise; push!(params, mgpcv.logNoise); end
    if domean;  append!(params, get_params(mgpcv.m)); end
    if kern; append!(params,  get_params(mgpcv.k)); end
    if beta; append!(params,  get_params(mgpcv.βkern)); end
    return params
end
function set_params!(mgpcv::MultiGPCovars, hyp::Vector{Float64}; 
                    noise::Bool=true, domean::Bool=true, kern::Bool=true, beta::Bool=true)
    i=1
    if noise
        mgpcv.logNoise = hyp[i]
        i+=1
    end
    if domean
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

function optimize!(mgpcv::MultiGPCovars; noise::Bool=true, domean::Bool=true, kern::Bool=true, beta::Bool=true, 
                    method=ConjugateGradient(), options=Optim.Options())
    cK_buffer = Array{Float64}(mgpcv.nobsv, mgpcv.nobsv)
    function mll(hyp::Vector{Float64})
        try
            set_params!(mgpcv, hyp; noise=noise, domean=domean, kern=kern, beta=beta)
            update_mll!(mgpcv)
            return -mgpcv.mll
        catch err
             if !all(isfinite.(hyp))
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
            set_params!(mgpcv, hyp; noise=noise, domean=domean, kern=kern, beta=beta)
            update_mll_and_dmll!(mgpcv, cK_buffer; noise=noise, domean=domean, kern=kern, beta=beta)
            grad[:] = -mgpcv.dmll
            return -mgpcv.mll
        catch err
             if !all(isfinite.(hyp))
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

    init = get_params(mgpcv;  noise=noise, domean=domean, kern=kern, beta=beta)  # Initial hyperparameter values
    func = Optim.OnceDifferentiable(mll, dmll!, mll_and_dmll!, init)
    results = Optim.optimize(func,init, method, options)
    set_params!(mgpcv, Optim.minimizer(results); noise=noise, domean=domean, kern=kern, beta=beta)
    mll(Optim.minimizer(results))
    return results
end

"""
    Obtain the full (block diagonal) spatial covariance matrix
"""
function spatial_cov(mgpcv::MultiGPCovars)
    cK = zeros(mgpcv.nobsv, mgpcv.nobsv)
    for key in mgpcv.groupKeys
        gp = mgpcv.mgp[key]
        slice = mgpcv.groupIndices[key]
        addcov!(view(cK, slice, slice), mgpcv.k, gp.X, gp.data)
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
    for key in mgpcv.groupKeys
        gp = mgpcv.mgp[key]
        slice = mgpcv.groupIndices[key]
        μ[slice] = mean(mgpcv.m,gp.X)
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
    precision = PDMats.X_invA_Xt(ΣY_β, mgpcv.D)
    for i in 1:mgpcv.p
        precision[i,i] += mgpcv.βkern.ℓ2
    end
    m = mean(mgpcv)
    βhat = (precision \ mgpcv.D) * (ΣY_β \ (mgpcv.y.-m))
    return βhat
end

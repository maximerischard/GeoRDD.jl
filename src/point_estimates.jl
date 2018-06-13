using Distributions: Normal, MultivariateNormal
using PDMats: AbstractPDMat
using GaussianProcesses: GPE, MatF64, mean, update_mll!, Mean

function inverse_variance(μ::AbstractVector, Σ::AbstractMatrix)
    n = size(μ)
    denom = sum(Σ \ ones(n))
    τhat = sum(Σ \ μ) / denom
    Vτhat = 1.0/denom
    τpost=Normal(τhat, √Vτhat)
    return τpost
end
    
# It really should be possible to write this function only once,
# but for some reason AbstractPDMat does not derive from
# AbstractMatrix
function inverse_variance(μ::AbstractVector, Σ::AbstractPDMat)
    n = size(μ)
    denom = sum(Σ \ ones(n))
    τhat = sum(Σ \ μ) / denom
    Vτhat = 1.0/denom
    τpost=Normal(τhat, √Vτhat)
    return τpost
end

function unweighted_mean(μ::AbstractVector, Σ::AbstractMatrix)
    n = length(μ)
    τhat = mean(μ)
    Vτhat = sum(Σ) / n^2
    τpost=Normal(τhat, √Vτhat)
    return τpost
end
function unweighted_mean(μ::AbstractVector, Σ::AbstractPDMat)
    return unweighted_mean(μ, full(Σ))
end

function weighted_mean(μ::AbstractVector, Σ::AbstractMatrix, w::AbstractVector)
    τhat = sum(μ .* w) / sum(w)
    Vτhat = dot(w, Σ*w) / sum(w)^2
    τpost=Normal(τhat, √Vτhat)
    return τpost
end
function weighted_mean(μ::AbstractVector, Σ::AbstractPDMat, w)
    return weighted_mean(μ, full(Σ), w)
end

# Benavoli and Mangili 2015
function get_pval(μ::Vector{Float64}, Σ::Matrix{Float64}, ϵ::Float64)    
    Σsvd = svdfact(Σ)
    λvec = Σsvd[:S] # sorted in decreasing order
    aboveϵ = (λvec ./ sum(λvec)) .> ϵ
    ν = max(sum(aboveϵ),1)
    Sabove = Σsvd[:S][aboveϵ]
    Σsvd[:S][:] = 1./Σsvd[:S] # invert high eigenvalues
    Σsvd[:S][!aboveϵ] = 0.0 # but remove low eigenvalues
    t = dot(μ, (full(Σsvd) * μ))
    nullhypo = Chisq(ν)
    pval = ccdf(nullhypo, t)
    return pval
end

function chisquare(gpT::GPE, gpC::GPE, X∂::Matrix, ϵ; verbose=false)
    extrap_T = predict_f(gpT, X∂; full_cov=true)
    extrap_C = predict_f(gpC, X∂; full_cov=true)
    μpost = extrap_T[1].-extrap_C[1]
    
    K∂C = cov(gpC.k, X∂, gpC.X)
    KC∂ = K∂C'
    KCC = gpC.cK
    
    KCT = cov(gpC.k, gpC.X, gpT.X)
    KTT = gpT.cK
    KT∂ = cov(gpT.k, gpT.X, X∂)
    K∂T = KT∂'
    
    K∂CT∂ = K∂C * (KCC \ KCT) * (KTT \ KT∂)
    
    Σ∂T12 = PDMats.whiten(KTT, KT∂)
#     Σ∂T = K∂T * (KTT \ KT∂)
    Σ∂T = Σ∂T12' * Σ∂T12
    
    Σ∂C12 = PDMats.whiten(KCC, KC∂)
    Σ∂C = Σ∂C12' * Σ∂C12
    
    chi2cov = Σ∂T + Σ∂C - (K∂CT∂ + K∂CT∂')
    if verbose
        evals = eigvals(Symmetric(chi2cov))
        print(evals)
        plt.semilogy(eigvals(Symmetric(chi2cov)))
        thresh = minimum(evals[evals ./ sum(evals) .> ϵ])
        plt.axhline(thresh, color="red")
    end
    return get_pval(μpost, chi2cov, ϵ)
end

function modifiable(gp::GPE)
    gp_copy = GPE(gp.m, gp.k, gp.logNoise, gp.nobsv,
        gp.X, copy(gp.y), gp.data,
        gp.dim, gp.cK, copy(gp.alpha),
        gp.mll, gp.mll, Float64[], Float64[])
    return gp_copy
end

function update_alpha!(gp::GPE)
    m = mean(gp.m,gp.X)
    gp.alpha = gp.cK \ (gp.y - m)
end

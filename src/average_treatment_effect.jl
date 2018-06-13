using Distributions: Normal, MultivariateNormal
using PDMats: AbstractPDMat
using GaussianProcesses: GPE, MatF64, mean, update_mll!, Mean

function inverse_variance(μ::AbstractVector, Σ::Union{AbstractMatrix,AbstractPDMat})
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

function update_alpha!(gp::GPE)
    m = mean(gp.m,gp.X)
    gp.alpha = gp.cK \ (gp.y - m)
end

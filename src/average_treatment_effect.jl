function inverse_variance(μ::AbstractVector, Σ::M) where {M<:Union{AbstractMatrix,AbstractPDMat}}
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

function weighted_mean(μ::AbstractVector, Σ::M, w::AbstractVector) where {M<:Union{AbstractMatrix,AbstractPDMat}}
    τhat = sum(μ .* w) / sum(w)
    Vτhat = dot(w, Σ*w) / sum(w)^2
    τpost=Normal(τhat, √Vτhat)
    return τpost
end
# function weighted_mean(μ::AbstractVector, Σ::AbstractPDMat, w)
    # return weighted_mean(μ, full(Σ), w)
# end


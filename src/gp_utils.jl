import GaussianProcesses: MatF64, VecF64, Mean, Kernel
import Base: copy


function predict_mu(gp::GPE, X::MatF64, cK::MatF64)
    n = size(X, 2)
    mu = mean(gp.m, X) + cK*gp.alpha        # Predictive mean
    return mu
end

function _predict_raw{M<:MatF64}(gp::GPE, X::M)
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = PDMats.whiten(gp.cK, cK')
    mu = predict_mu(gp, X, cK)             # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck     # Predictive covariance
    return mu, Sigma_raw
end

""" Copy of a Gaussian process that shares X and cK but allows
    modification of the outcomes Y. This is useful for bootstrapping.
"""
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
function mll(gp::GPE)
    μ = mean(gp.m,gp.X)
    return -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

function addcov!(cK::AbstractMatrix{Float64}, k::Kernel, X::AbstractMatrix{Float64}, data::KernelData)
    dim, nobsv = size(X)
    @assert size(cK, 1) == nobsv
    @assert size(cK, 2) == nobsv
    @inbounds for j in 1:nobsv
        for i in 1:j
            cK[i,j] += cov_ij(k, X, data, i, j, dim)
            cK[j,i] = cK[i,j]
        end
    end
end
function add_diag!(mat::AbstractMatrix, d::Real)
    n = size(mat, 1)
    if n!=size(mat, 2)
        throw("Matrix is not square")
    end
    @inbounds for i in 1:n
        mat[i,i] += d
    end
    return mat
end
function update_chol!(pd::PDMats.PDMat)
    mat = pd.mat
    chol_buffer = pd.chol.factors
    copy!(chol_buffer, mat)
    chol = cholfact!(Symmetric(chol_buffer))
    return PDMats.PDMat(mat, chol)
end

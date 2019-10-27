function predict_mu(gp::GPE, X::AbstractMatrix, cK::AbstractMatrix)
    n = size(X, 2)
    mu = mean(gp.mean, X) + cK*gp.alpha        # Predictive mean
    return mu
end

""" Copy of a Gaussian process that shares X and cK but allows
    modification of the outcomes Y. This is useful for bootstrapping.
"""
function modifiable(gp::GPE)
    gp_copy = GPE(gp.x, copy(gp.y), gp.mean, gp.kernel, 
                  gp.logNoise, gp.covstrat, gp.data, gp.cK)
    return gp_copy
end
function update_alpha!(gp::GPE)
    m = mean(gp.mean,gp.x)
    gp.alpha = gp.cK \ (gp.y - m)
end
function mll(gp::GPE)
    μ = mean(gp.mean,gp.x)
    return -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobs*log(2π)/2.0 # Marginal log-likelihood
end

function addcov!(cK::AbstractMatrix{Float64}, kernel::Kernel, X::AbstractMatrix{Float64}, data::KernelData)
    dim, nobs = size(X)
    @assert size(cK, 1) == nobs
    @assert size(cK, 2) == nobs
    @inbounds for j in 1:nobs
        for i in 1:j
            cK[i,j] += cov_ij(kernel, X, X, data, i, j, dim)
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
    Σbuffer = mat(pd)
    Σbuffer, chol = make_posdef!(Σbuffer, cholfactors(pd))
    new_pd = wrap_cK(pd, Σbuffer, chol)
    return new_pd
end

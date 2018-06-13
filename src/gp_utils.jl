import GaussianProcesses: MatF64, VecF64, Mean, Kernel

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


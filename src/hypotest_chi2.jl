using PDMats: PDMat

function chistat(gpT::GPE, gpC::GPE, Xb::AbstractMatrix)
    pred_T = _predict_raw(gpT, Xb)
    pred_C = _predict_raw(gpC, Xb)
    μ = pred_T[1].-pred_C[1]
    Σ = pred_T[2]+pred_C[2]
    return dot(μ, Σ \ μ)
end
function chistat(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, 
                 Σcliff::PDMat, cK_T::AbstractMatrix, cK_C::AbstractMatrix)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μ = μT - μC
    return dot(μ, Σcliff \ μ)
end

function sim_chi_null!(
        gpT::GPE, gpC::GPE, gpNull::GPE, 
        treat::BitVector, Xb::AbstractMatrix, 
        Σcliff::PDMat, cK_T::AbstractMatrix, cK_C::AbstractMatrix)
    Ysim = prior_rand(gpNull)

    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return chistat(gpT, gpC, Xb, Σcliff, cK_T, cK_C)
end
function nsim_chi(gpT::GPE, gpC::GPE, gpNull::GPE, Xb::AbstractMatrix, nsim::Int)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)

    _, Σcliff = cliff_face(gpT, gpC, Xb)
    cK_T = cov(gpT.kernel, Xb, gpT.x)
    cK_C = cov(gpC.kernel, Xb, gpC.x)

    treat = BitVector(gpNull.nobs)
    treat[:] = false
    treat[1:gpT.nobs] = true
    kernel = gpT_mod.kernel
    mT = gpT_mod.mean
    mC = gpC_mod.mean
    t_sims = [sim_chi_null!(gpT_mod, gpC_mod, gpNull, treat, Xb, Σcliff, cK_T, cK_C) 
        for _ in 1:nsim];
    return t_sims
end

function boot_chi2test(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, nsim::Int)
    gpNull = make_null(gpT, gpC)
    chi_obs = chistat(gpT, gpC, Xb)
    chi_sims = GeoRDD.nsim_chi(gpT, gpC, gpNull, Xb, nsim)
    return mean(chi_sims .> chi_obs)
end


function placebo_chi(angle::Float64, X::AbstractMatrix, Y::Vector,
                 kern::Kernel, mean::Mean, logNoise::Float64, 
                 nsim::Int)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],   Y[left], mean, kern, logNoise)
    gp_right = GPE(X[:,.!left], Y[.!left], mean, kern, logNoise)
    Xb = placebo_sentinels(angle, shift, X, 100)
    pval = boot_chi2test(gp_left, gp_right, Xb, nsim)
    return pval
end
function placebo_chi(angle::Float64, gp::GPE, nsim::Int; kwargs...)
    return placebo_chi(angle, gp.x, gp.y, gp.kernel, gp.mean, gp.logNoise, nsim; kwargs...)
end

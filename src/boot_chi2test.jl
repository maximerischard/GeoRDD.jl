function chistat(gpT::GPE, gpC::GPE, Xb::MatF64)
    pred_T = _predict_raw(gpT, Xb)
    pred_C = _predict_raw(gpC, Xb)
    μ = pred_T[1].-pred_C[1]
    Σ = pred_T[2]+pred_C[2]
    return dot(μ, Σ \ μ)
end
function chistat(gpT::GPE, gpC::GPE, Xb::MatF64, 
                                     Σcliff::PDMat, cK_T::MatF64, cK_C::MatF64)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μ = μT - μC
    return dot(μ, Σcliff \ μ)
end


function sim_chi_null!(
        gpT::GPE, gpC::GPE, gpNull::GPE, 
        treat::BitVector, Xb::MatF64, 
        Σcliff::PDMat, cK_T::MatF64, cK_C::MatF64; update_mean::Bool=false)
    n = gpNull.nobsv
    null = MultivariateNormal(zeros(n), gpNull.cK)
    Ysim = rand(null)

    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]

    if update_mean
        gpT.m = mT = MeanConst(mean(gpT.y))
        gpC.m = mC = MeanConst(mean(gpC.y))
    end

    update_alpha!(gpT)
    update_alpha!(gpC)

    return chistat(gpT, gpC, Xb, Σcliff, cK_T, cK_C)
end
function nsim_chi(gpT::GPE, gpC::GPE, Xb::MatF64, nsim::Int; update_mean::Bool=false)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    yNull = [gpT.y; gpC.y]
    gpNull = GPE([gpT.X gpC.X], yNull, MeanConst(mean(yNull)), gpT.k, gpT.logNoise)

    _, Σcliff = cliff_face(gpT, gpC, Xb)
    cK_T = cov(gpT.k, Xb, gpT.X)
    cK_C = cov(gpC.k, Xb, gpC.X)

    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    k = gpT_mod.k
    mT = gpT_mod.m
    mC = gpC_mod.m
    t_sims = [sim_chi_null!(gpT_mod, gpC_mod, gpNull, treat, Xb, Σcliff, cK_T, cK_C; update_mean=update_mean) 
        for _ in 1:nsim];
    return t_sims
end

function boot_chi2test(gpT::GPE, gpC::GPE, Xb::MatF64, nsim::Int; update_mean::Bool=false)
    chi_obs = chistat(gpT, gpC, Xb)
    chi_sims = GeoRDD.nsim_chi(gpT, gpC, Xb, nsim; update_mean=update_mean)
    return mean(chi_sims .> chi_obs)
end


function placebo_chi(angle::Float64, X::MatF64, Y::Vector,
                 kern::Kernel, logNoise::Float64, 
                 nsim::Int; update_mean::Bool=false)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],
                   Y[left],
                   MeanConst(mean(Y[left])),
                   kern,
                   logNoise)
    gp_right = GPE(X[:, .!left],
                   Y[.!left],
                   MeanConst(mean(Y[.!left])),
                   kern,
                   logNoise)
    Xb = placebo_sentinels(angle, shift, X, 100)
    pval = boot_chi2test(gp_left, gp_right, Xb, nsim; update_mean=update_mean)
    return pval
end

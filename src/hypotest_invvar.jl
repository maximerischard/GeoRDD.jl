function pval_invvar(gpT::GPE, gpC::GPE, Xb::MatF64)
    μ, Σ = cliff_face(gpT, gpC, Xb)
    invvar = inverse_variance(μ, Σ)
    pval_invvar = 2*min(cdf(invvar, 0.0), ccdf(invvar, 0.0))
    return pval_invvar
end

""" 
With pre-computed cliff-face variance (for simulations with fixed positions).
"""
function pval_invvar(gpT::GPE, gpC::GPE, Xb::MatF64, Σcliff::PDMat, cK_T::MatF64, cK_C::MatF64)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μ = μT - μC
    invvar = inverse_variance(μ, Σcliff)
    pval_invvar = 2*min(cdf(invvar, 0.0), ccdf(invvar, 0.0))
    return pval_invvar
end

function sim_invvar!(gpT::GPE, gpC::GPE, gpNull::GPE, 
            treat::BitVector, Xb::MatF64)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return pval_invvar(gpT, gpC, Xb)
end

function sim_invvar!(gpT::GPE, gpC::GPE, gpNull::GPE, 
            treat::BitVector, Xb::MatF64, 
            Σcliff::PDMat, cK_T::MatF64, cK_C::MatF64)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return pval_invvar(gpT, gpC, Xb, Σcliff, cK_T, cK_C)
end

function nsim_invvar_pval(gpT::GPE, gpC::GPE, Xb::MatF64, nsim::Int)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    _, Σcliff = cliff_face(gpT, gpC, Xb)
    cK_T = cov(gpT.k, Xb, gpT.X)
    cK_C = cov(gpC.k, Xb, gpC.X)
    gpNull = make_null(gpT, gpC)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    pval_sims = Float64[sim_invvar!(gpT_mod, gpC_mod, gpNull, treat, Xb, Σcliff, cK_T, cK_C) 
        for _ in 1:nsim];
    return pval_sims
end

function boot_invvar(gpT::GPE, gpC::GPE, Xb::MatF64, nsim::Int)
    pval_obs = pval_invvar(gpT, gpC, Xb)
    pval_sims = nsim_invvar_pval(gpT, gpC, Xb, nsim)
    return mean(pval_obs .< pval_sims)
end

#============================================
    ANALYTIC INSTEAD OF BOOTSTRAP CALIBRATION
=============================================#
function pval_invvar_calib(gpT::GPE, gpC::GPE, Xb::Matrix)
    predT_b = GaussianProcesses.predict_f(gpT, Xb; full_cov=true)
    predC_b = GaussianProcesses.predict_f(gpC, Xb; full_cov=true)
    μb = predT_b[1].-predC_b[1]
    n = size(μb)
    Σb = predT_b[2]+predC_b[2]
    
    KbC = cov(gpC.k, Xb, gpC.X)
    KCb = KbC'
    KCC = gpC.cK
    
    KCT = cov(gpC.k, gpC.X, gpT.X)
    KTT = gpT.cK
    KTb = cov(gpT.k, gpT.X, Xb)
    KbT = KTb'

    AT_c = KTT \ KTb
    AC_c = KCC \ KCb
    AT = AT_c'
    AC = AC_c'
    cov_μδ = AT*full(KTT)*AT_c + 
             AC*full(KCC)*AC_c - 
             AC*full(KCT)*AT_c - 
             AT*full(KCT)'*AC_c
    
    cov_μτ = sum((Σb \ cov_μδ) * (Σb \ ones(n)))
    null = Normal(0.0, √cov_μτ)
    
    μτ_numer = sum(Σb \ μb) # numerator only
    
    pval = 2*ccdf(null, abs(μτ_numer))
    return pval
end
function pval_invvar_calib(gpT::GPE, gpC::GPE, Xb::Matrix, Σcliff::PDMat, cK_T::MatF64, cK_C::MatF64, KCT::MatF64)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μb = μT - μC
    n = size(μb)
    Σb = Σcliff
    
    KbC = cK_C
    KCb = KbC'
    KCC = gpC.cK
    
    KTT = gpT.cK
    KbT = cK_T
    KTb = KbT'

    AT_c = KTT \ KTb
    AC_c = KCC \ KCb
    AT = AT_c'
    AC = AC_c'
    cov_μδ = AT*full(KTT)*AT_c +
             AC*full(KCC)*AC_c -
             AC*full(KCT)*AT_c -
             AT*full(KCT)'*AC_c
    
    cov_μτ = sum((Σb \ cov_μδ) * (Σb \ ones(n)))
    null = Normal(0.0, √cov_μτ)
    
    μτ_numer = sum(Σb \ μb) # numerator only
    
    pval = 2*ccdf(null, abs(μτ_numer))
    return pval
end
function sim_invvar_calib!(gpT::GPE, gpC::GPE, gpNull::GPE, 
            treat::BitVector, Xb::MatF64)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    GeoRDD.update_alpha!(gpT)
    GeoRDD.update_alpha!(gpC)

    return pval_invvar_calib(gpT, gpC, Xb)
end
function nsim_invvar_calib(gpT::GPE, gpC::GPE, Xb::MatF64, nsim::Int)
    gpT_mod = GeoRDD.modifiable(gpT)
    gpC_mod = GeoRDD.modifiable(gpC)
    gpNull = make_null(gpT_mod, gpC_mod)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    pval_sims = Float64[sim_invvar_calib!(gpT_mod, gpC_mod, gpNull, treat, Xb) 
        for _ in 1:nsim];
    return pval_sims
end

function placebo_invvar(angle::Float64, X::MatF64, Y::Vector,
                        kern::Kernel, m::Mean, logNoise::Float64)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],   Y[left],   m, kern, logNoise)
    gp_right = GPE(X[:,.!left], Y[.!left], m, kern, logNoise)
    Xb = placebo_sentinels(angle, shift, X, 100)
    pval = pval_invvar_calib(gp_left, gp_right, Xb)
    return pval
end
function placebo_invvar(angle::Float64, gp::GPE; kwargs...)
    return placebo_invvar(angle, gp.X, gp.y, gp.k, gp.m, gp.logNoise; kwargs...)
end

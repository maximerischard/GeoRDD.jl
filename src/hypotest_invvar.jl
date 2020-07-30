function pval_invvar_uncalib(gpT::GPE, gpC::GPE, Xb::AbstractMatrix; nugget=1e-10)
    μ, Σ = cliff_face(gpT, gpC, Xb)
    invvar = inverse_variance(μ, Σ; nugget=nugget)
    pval_invvar = 2*min(cdf(invvar, 0.0), ccdf(invvar, 0.0))
    return pval_invvar
end

"""
With pre-computed cliff-face variance (for simulations with fixed positions).
"""
function pval_invvar_uncalib(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, Σcliff::PDMat,
                             cK_T::AbstractMatrix, cK_C::AbstractMatrix)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μ = μT - μC
    invvar = inverse_variance(μ, Σcliff; nugget=NaN) # no nugget because PDMat
    pval_invvar = 2*min(cdf(invvar, 0.0), ccdf(invvar, 0.0))
    return pval_invvar
end

function sim_invvar_uncalib!(gpT::GPE, gpC::GPE, gpNull::GPE,
            treat::BitVector, Xb::AbstractMatrix; nugget=1e-10)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return pval_invvar_uncalib(gpT, gpC, Xb; nugget=nugget)
end

function sim_invvar_uncalib!(gpT::GPE, gpC::GPE, gpNull::GPE,
            treat::BitVector, Xb::AbstractMatrix,
            Σcliff::PDMat, cK_T::AbstractMatrix, cK_C::AbstractMatrix)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return pval_invvar_uncalib(gpT, gpC, Xb, Σcliff, cK_T, cK_C)
end

function nsim_invvar_pval_uncalib(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, nsim::Int; nugget=1e-10)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    _, Σcliff = cliff_face(gpT, gpC, Xb)
    Σraw, chol = make_posdef!(add_nugget(Σcliff, nugget))
    PDcliff = PDMat(Σraw, chol)
    cK_T = cov(gpT.kernel, Xb, gpT.x)
    cK_C = cov(gpC.kernel, Xb, gpC.x)
    gpNull = make_null(gpT, gpC)
    treat = BitVector(undef, gpNull.nobs)
    treat[:] .= false
    treat[1:gpT.nobs] .= true
    pval_sims = Float64[sim_invvar_uncalib!(gpT_mod, gpC_mod, gpNull, treat, Xb, PDcliff, cK_T, cK_C)
        for _ in 1:nsim];
    return pval_sims
end

function boot_invvar(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, nsim::Int; nugget=1e-10)
    pval_obs = pval_invvar_uncalib(gpT, gpC, Xb; nugget=nugget)
    pval_sims = nsim_invvar_pval_uncalib(gpT, gpC, Xb, nsim; nugget=nugget)
    return mean(pval_sims .< pval_obs)
end

#============================================
    ANALYTIC INSTEAD OF BOOTSTRAP CALIBRATION
=============================================#
function pval_invvar_calib(gpT::GPE, gpC::GPE, Xb::Matrix; nugget=1e-10)
    μb, Σb = cliff_face(gpT::GPE, gpC::GPE, Xb::Matrix)
    n = size(μb, 1)
    Σb = add_nugget(Σb, nugget)


    KbC = cov(gpC.kernel, Xb, gpC.x)
    KCb = Matrix(KbC')
    KCC = gpC.cK

    KCT = cov(gpC.kernel, gpC.x, gpT.x)
    KTT = gpT.cK
    KTb = cov(gpT.kernel, gpT.x, Xb)
    KbT = KTb'

    WT_c = KTT \ KTb
    WC_c = KCC \ KCb
    WT = WT_c'
    WC = WC_c'
    cov_μδ = WT*Matrix(KTT)*WT_c +
             WC*Matrix(KCC)*WC_c -
             WC*Matrix(KCT)*WT_c -
             WT*Matrix(KCT)'*WC_c

    cov_μτ = sum((Σb \ cov_μδ) * (Σb \ ones(n)))
    null = Normal(0.0, √cov_μτ)

    μτ_numer = sum(Σb \ μb) # numerator only

    pval = 2*ccdf(null, abs(μτ_numer))
    return pval
end
function pval_invvar_calib(gpT::GPE, gpC::GPE, Xb::Matrix, Σcliff::PDMat, cK_T::AbstractMatrix, cK_C::AbstractMatrix, KCT::AbstractMatrix)
    μT = predict_mu(gpT, Xb, cK_T)
    μC = predict_mu(gpC, Xb, cK_C)
    μb = μT - μC
    n = size(μb, 1)
    Σb = Σcliff

    KbC = cK_C
    KCb = KbC'
    KCC = gpC.cK

    KTT = gpT.cK
    KbT = cK_T
    KTb = KbT'

    WT_c = KTT \ KTb
    WC_c = KCC \ KCb
    WT = WT_c'
    WC = WC_c'
    cov_μδ = WT*Matrix(KTT)*WT_c
             WC*Matrix(KCC)*WC_c -
             WC*Matrix(KCT)*WT_c -
             WT*Matrix(KCT)'*WC_c

    cov_μτ = sum((Σb \ cov_μδ) * (Σb \ ones(n)))
    null = Normal(0.0, √cov_μτ)

    μτ_numer = sum(Σb \ μb) # numerator only

    pval = 2*ccdf(null, abs(μτ_numer))
    return pval
end
function sim_invvar_calib!(gpT::GPE, gpC::GPE, gpNull::GPE,
            treat::BitVector, Xb::AbstractMatrix)
    Ysim = prior_rand(gpNull)

    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]

    GeoRDD.update_alpha!(gpT)
    GeoRDD.update_alpha!(gpC)

    return pval_invvar_calib(gpT, gpC, Xb)
end
function nsim_invvar_calib(gpT::GPE, gpC::GPE, Xb::AbstractMatrix, nsim::Int)
    gpT_mod = GeoRDD.modifiable(gpT)
    gpC_mod = GeoRDD.modifiable(gpC)
    gpNull = make_null(gpT_mod, gpC_mod)
    treat = BitVector(undef, gpNull.nobs)
    treat[:] .= false
    treat[1:gpT.nobs] .= true
    pval_sims = Float64[sim_invvar_calib!(gpT_mod, gpC_mod, gpNull, treat, Xb)
        for _ in 1:nsim];
    return pval_sims
end

function placebo_invvar(angle::Float64, X::AbstractMatrix, Y::Vector,
                        kern::Kernel, mean::Mean, logNoise::Float64)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],   Y[left],   mean, kern, logNoise)
    gp_right = GPE(X[:,.!left], Y[.!left], mean, kern, logNoise)
    Xb = placebo_sentinels(angle, shift, X, 100)
    pval = pval_invvar_calib(gp_left, gp_right, Xb)
    return pval
end
function placebo_invvar(angle::Float64, gp::GPE; kwargs...)
    logNoise = convert(Float64, gp.logNoise)
    return placebo_invvar(angle, gp.x, gp.y, gp.kernel, gp.mean, logNoise; kwargs...)
end

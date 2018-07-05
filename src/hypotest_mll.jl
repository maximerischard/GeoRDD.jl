function sim_logP!(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector)
    Ysim = prior_rand(gpNull)
    
    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]
    gpNull.y = Ysim
    
    update_alpha!(gpT)
    update_alpha!(gpC)
    update_alpha!(gpNull)
    
    gpT.mll = mll(gpT)
    gpC.mll = mll(gpC)
    gpNull.mll = mll(gpNull)
    
    mll_altv = gpT.mll + gpC.mll
    mll_null = gpNull.mll
    return mll_null, mll_altv
end

function nsim_logP(gpT::GPE, gpC::GPE, gpNull::GPE, nsim::Int)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    mll_sims = [sim_logP!(gpT_mod, gpC_mod, gpNull, treat) 
        for _ in 1:nsim];
    return mll_sims
end

function boot_mlltest(gpT::GPE, gpC::GPE, nsim::Int)
    mll_alt = gpT.mll + gpC.mll
    
    gpNull = make_null(gpT, gpC)
    mll_null = gpNull.mll
    mll_sims = GeoRDD.nsim_logP(gpT, gpC, gpNull, nsim)
    
    mll_sim_null = [sim[1] for sim in mll_sims]
    mll_sim_altv = [sim[2] for sim in mll_sims]
    
    Δmll_obs = mll_alt - mll_null
    Δmll_sim = mll_sim_altv .- mll_sim_null
    return mean(Δmll_sim .> Δmll_obs)
end

function placebo_mll(angle::Float64, X::Matrix, Y::Vector, 
                 kern::Kernel, m::Mean, logNoise::Float64, 
                 nsim::Int)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],  Y[left],  m, kern, logNoise)
    gp_right = GPE(X[:,.!left], Y[.!left], m, kern, logNoise)
    pval = boot_mlltest(gp_left, gp_right, nsim)
    return pval
end
function placebo_mll(angle::Float64, gp::GPE, nsim::Int; kwargs...)
    return placebo_mll(angle, gp.X, gp.y, gp.k, gp.m, gp.logNoise, nsim; kwargs...)
end

function mll(gp::GPE)
    μ = mean(gp.m,gp.X)
    return -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

function sim_logP!(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector; update_mean::Bool=false)
    n = gpNull.nobsv
    null = MultivariateNormal(zeros(n), gpNull.cK)
    Ysim = rand(null)
    
    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]
    gpNull.y = Ysim
    
    if update_mean
        gpT.m = MeanConst(mean(gpT.y))
        gpC.m = MeanConst(mean(gpC.y))
        gpNull.m = MeanConst(mean(gpNull.y))
    end
        
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

function nsim_logP(gpT::GPE, gpC::GPE, 
                  nsim::Int; update_mean::Bool=false)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    yNull = [gpT_mod.y; gpC_mod.y]
    gpNull = GPE([gpT.X gpC.X], yNull, MeanConst(mean(yNull)), gpT.k, gpT.logNoise)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    mll_sims = [sim_logP!(gpT_mod, gpC_mod, gpNull, treat; update_mean=update_mean) 
        for _ in 1:nsim];
    return mll_sims
end

function data_logP(gpT::GPE, gpC::GPE)
    yNull = [gpT.y; gpC.y]
    gpNull = GPE([gpT.X gpC.X], yNull, MeanConst(mean(yNull)), gpT.k, gpT.logNoise)
    update_mll!(gpNull)
    return gpNull.mll, gpT.mll+gpC.mll
end

function boot_mlltest(gpT::GPE, gpC::GPE, nsim::Int; update_mean::Bool=false)
    mll_alt = gpT.mll + gpC.mll
    
    yNull = [gpT.y; gpC.y]
    gpNull = GPE([gpT.X gpC.X], yNull, MeanConst(mean(yNull)), gpT.k, gpT.logNoise)
    
    mll_null = gpNull.mll
    mll_sims = GeoRDD.nsim_logP(gpT, gpC, nsim; update_mean=update_mean)
    
    mll_sim_null = [sim[1] for sim in mll_sims]
    mll_sim_altv = [sim[2] for sim in mll_sims]
    
    Δmll_obs = mll_alt - mll_null
    Δmll_sim = mll_sim_altv .- mll_sim_null
    return mean(Δmll_sim .> Δmll_obs)
end

function placebo_mll(angle::Float64, X::Matrix, Y::Vector, 
                 kern::Kernel, logNoise::Float64, 
                 nsim::Int; update_mean::Bool=false)
    shift = shift_for_even_split(angle, X)
    left = left_points(angle, shift, X)
    gp_left  = GPE(X[:,left],  Y[left],  MeanConst(mean(Y[left])),  kern, logNoise)
    gp_right = GPE(X[:,.!left], Y[.!left], MeanConst(mean(Y[.!left])), kern, logNoise)
    pval = boot_mlltest(gp_left, gp_right, nsim; update_mean=update_mean)
    return pval
end

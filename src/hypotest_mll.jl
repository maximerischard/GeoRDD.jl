function sim_logP!(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector; update_mean::Bool=false)
    # μ = mean(gpNull.m, gpNull.X)
    μ = zeros(gpNull.nobsv)
    null = MultivariateNormal(μ, gpNull.cK)
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

function nsim_logP(gpT::GPE, gpC::GPE, gpNull::GPE, nsim::Int; update_mean::Bool=false)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    mll_sims = [sim_logP!(gpT_mod, gpC_mod, gpNull, treat; update_mean=update_mean) 
        for _ in 1:nsim];
    return mll_sims
end

function boot_mlltest(gpT::GPE, gpC::GPE, nsim::Int; update_mean::Bool=false)
    mll_alt = gpT.mll + gpC.mll
    
    yNull = [gpT.y; gpC.y]
    xNull = [gpT.X gpC.X]
    kNull = gpT.k
    mNull = MeanConst(mean(yNull))
    gpNull = GPE(xNull, yNull, mNull, kNull, gpT.logNoise)
    
    mll_null = gpNull.mll
    mll_sims = GeoRDD.nsim_logP(gpT, gpC, gpNull, nsim; update_mean=update_mean)
    
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
function placebo_mll(angle::Float64, gp::GPE, nsim::Int; kwargs...)
    return placebo_mll(angle, gp.X, gp.y, gp.k, gp.logNoise, nsim; kwargs...)
end

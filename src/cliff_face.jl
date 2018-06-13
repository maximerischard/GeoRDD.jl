import GaussianProcesses: GPE, MatF64, predict_f

function cliff_face(gpT::GPE, gpC::GPE, sentinels::MatF64)
    pred_T = predict_f(gpT, sentinels; full_cov=true)
    pred_C = predict_f(gpC, sentinels; full_cov=true)
    μposterior = pred_T[1].-pred_C[1]
    Σposterior = pred_T[2]+pred_C[2]
    return μposterior, Σposterior
end

function sim_cliff(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector, X∂::MatF64; update_mean::Bool=false)
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

    return cliff_face(gpT, gpC, X∂)
end
function nsim_cliff(gpT::GPE, gpC::GPE, X∂::MatF64, nsim::Int; update_mean::Bool=false)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    yNull = [gpT.y; gpC.y]
    gpNull = GPE([gpT.X gpC.X], yNull, MeanConst(mean(yNull)), gpT.k, gpT.logNoise)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    k = gpT_mod.k
    mT = gpT_mod.m
    mC = gpC_mod.m
    cliff_sims = [sim_cliff(gpT_mod, gpC_mod, gpNull, treat, X∂; update_mean=update_mean) 
        for _ in 1:nsim];
    return cliff_sims
end

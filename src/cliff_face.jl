import GaussianProcesses: GPE, MatF64, predict_f

function cliff_face(gpT::GPE, gpC::GPE, sentinels::MatF64)
    pred_T = predict_f(gpT, sentinels; full_cov=true)
    pred_C = predict_f(gpC, sentinels; full_cov=true)
    μposterior = pred_T[1].-pred_C[1]
    Σposterior = pred_T[2]+pred_C[2]
    return μposterior, Σposterior
end

function sim_cliff(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector, X∂::MatF64)
    μ = mean(gpNull.m, gpNull.X)
    null = MultivariateNormal(μ, gpNull.cK)
    Ysim = rand(null)

    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return cliff_face(gpT, gpC, X∂)
end
function nsim_cliff(gpT::GPE, gpC::GPE, X∂::MatF64, nsim::Int)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    yNull = [gpT.y; gpC.y]
    mNull = gpT.m
    gpNull = GPE([gpT.X gpC.X], yNull, mNull, gpT.k, gpT.logNoise)
    treat = BitVector(gpNull.nobsv)
    treat[:] = false
    treat[1:gpT.nobsv] = true
    k = gpT_mod.k
    mT = gpT_mod.m
    mC = gpC_mod.m
    cliff_sims = [sim_cliff(gpT_mod, gpC_mod, gpNull, treat, X∂) 
                  for _ in 1:nsim];
    return cliff_sims
end

import GaussianProcesses: GPE, AbstractMatrix, predict_f

function cliff_face(gpT::GPE, gpC::GPE, sentinels::AbstractMatrix)
    pred_T = predict_f(gpT, sentinels; full_cov=true)
    pred_C = predict_f(gpC, sentinels; full_cov=true)
    μposterior = pred_T[1].-pred_C[1]
    Σposterior = pred_T[2]+pred_C[2]
    return μposterior, Σposterior
end

function sim_cliff(gpT::GPE, gpC::GPE, gpNull::GPE, treat::BitVector, X∂::AbstractMatrix)
    μ = mean(gpNull.m, gpNull.x)
    null = MultivariateNormal(μ, gpNull.cK)
    Ysim = rand(null)

    gpT.y = Ysim[treat]
    gpC.y = Ysim[.!treat]

    update_alpha!(gpT)
    update_alpha!(gpC)

    return cliff_face(gpT, gpC, X∂)
end
function nsim_cliff(gpT::GPE, gpC::GPE, X∂::AbstractMatrix, nsim::Int)
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    yNull = [gpT.y; gpC.y]
    mNull = gpT.m
    gpNull = GPE([gpT.x gpC.x], yNull, mNull, gpT.kernel, gpT.logNoise)
    treat = BitVector(gpNull.nobs)
    treat[:] = false
    treat[1:gpT.nobs] = true
    mT = gpT_mod.m
    mC = gpC_mod.m
    cliff_sims = [sim_cliff(gpT_mod, gpC_mod, gpNull, treat, X∂) 
                  for _ in 1:nsim];
    return cliff_sims
end

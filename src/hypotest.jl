function make_null(gpT::GPE, gpC::GPE, kNull::Kernel, mNull::Mean, logNoise::Float64)
    yNull = [gpT.y; gpC.y]
    xNull = [gpT.X gpC.X]
    gpNull = GPE(xNull, yNull, mNull, kNull, logNoise)
end
function make_null(gpT::GPE, gpC::GPE)
    # copy parameters from treatment GP
    kNull = gpT.k
    mNull = gpT.m
    logNoise = gpT.logNoise
    return make_null(gpT, gpC, kNull, mNull, logNoise)
end
function prior_rand(gp::GPE)
    μ = mean(gp.m, gp.X)
    mvn = MultivariateNormal(μ, gp.cK)
    Ysim = rand(mvn)
    return Ysim
end

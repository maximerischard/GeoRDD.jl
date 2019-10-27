function make_null(gpT::GPE, gpC::GPE, kNull::Kernel, mNull::Mean, logNoise::Float64)
    yNull = [gpT.y; gpC.y]
    xNull = [gpT.x gpC.x]
    gpNull = GPE(xNull, yNull, mNull, kNull, logNoise)
end
function make_null(gpT::GPE, gpC::GPE)
    # copy parameters from treatment GP
    kNull = gpT.kernel
    mNull = gpT.mean
    logNoise = convert(Float64, gpT.logNoise)
    return make_null(gpT, gpC, kNull, mNull, logNoise)
end
function prior_rand(gp::GPE)
    μ = mean(gp.mean, gp.x)
    mvn = MultivariateNormal(μ, gp.cK)
    Ysim = rand(mvn)
    return Ysim
end

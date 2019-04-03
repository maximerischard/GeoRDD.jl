using DataFrames
using StatsModels
using LibGEOS
using GaussianProcesses
using Distributions
using Random
using Optim
using Printf
using LinearAlgebra

function run_simulation()
    Random.seed!(1) # for replicability
    n = 300
    X = rand(2, n)
    f_X = sin.(X[1,:] * 2.0) .+ 3.0 .* X[2,:].^2.0
    m = -0.8 # mean offset
    τ = 0.3 # constant treatment effect
    treatment_radius = 0.8
    covarA = randn(n)
    covarB = randn(n)
    categ = rand(["A", "B", "C"], n) # random cateogry
    treat = sqrt.(X[1,:].^2 + X[2,:].^2) .< treatment_radius

    # outcome is: smooth surface + noise + treatment effect
    Y = m .+ # mean offset
        f_X .+ # smooth surface
        0.3*randn(n) .+ # noise
        τ .* treat .+ # treatment effect
        covarA .* 0.1 .+ # effect of real-valued covariate
        (categ.=="B") .* 0.2 # raise category B a little bit
    border_X1 = treatment_radius .* cos.(range(0,stop=π/2,length=1000))
    border_X2 = treatment_radius .* sin.(range(0,stop=π/2,length=1000))
    border_geo = LibGEOS.LineString([[border_X1[i], border_X2[i]] for i in 1:n])
    geordd_df = DataFrame(
        X1 = X[1, :],
        X2 = X[2, :],
        outcome = Y,
        region = treat, # treatment indicator
        covarA = covarA,
        covarB = covarB,
        categ = categ
        )
    categorical!(geordd_df, :categ)
    categorical!(geordd_df, :region)
    fmla = @formula(outcome ~ GP(X1, X2) | region + covarA + covarB + categ)
    geordd = GeoRDD.regions_from_dataframe(fmla, geordd_df)
    # choose a Gaussian process kernel from the GaussianProcesses.jl package
    k_se = SEIso(log(0.5), log(1.0)) # Squared Exponential spatial kernel
    k_m = Const(log(20.0))           # constant kernel for the mean parameters
    βkern = LinIso(log(1.0))         # linear kernel for the other covariates
    logNoise = 1.0
    # create the Gaussian process object from the regional data:
    mgpcv = GeoRDD.MultiGPCovars(geordd, k_se+k_m, βkern, logNoise)
    # optimize the hyperparameters
    @time opt_output=optimize!(
            mgpcv, 
            noise=true,
            kern=true,
            domean=false,
            beta=true,
            options=Optim.Options(
                show_trace=false,
                iterations=1000,
                x_tol=1e-8,
                f_tol=1e-10),
        )
    # extract posterior mean of linear regression coefficients
    βhat = GeoRDD.postmean_β(mgpcv)
    # extract residuals
    residuals_rd = copy(geordd)
    residuals_rd[true]  = GeoRDD.residuals_data(geordd[true],  βhat) # treatment region
    residuals_rd[false] = GeoRDD.residuals_data(geordd[false], βhat) # control region
    # Gaussian processes fitted to residuals:
    resid_GP_dict = GeoRDD.GPRealisations(residuals_rd, mgpcv.kernel, mgpcv.logNoise)
    # obtain posterior treatment effect along border
    T,C = true, false
    sentinels = GeoRDD.sentinels(border_geo, 100)
    μpost, Σpost = GeoRDD.cliff_face(resid_GP_dict[T], 
                                     resid_GP_dict[C],
                                     sentinels)
    τ_inv=GeoRDD.inverse_variance(μpost, Σpost)
    # check the credible interval is not ridiculous
    @test τ > quantile(τ_inv, 0.001)
    @test τ < quantile(τ_inv, 0.999)
    kernspace = mgpcv.kernel.kleft
    ℓ = sqrt(kernspace.ℓ2)
    maxdist = 2ℓ
    τ_proj = GeoRDD.proj_estimator(resid_GP_dict[T], 
                                   resid_GP_dict[C], 
                                   border_geo,
                                   2ℓ)
    @test τ > quantile(τ_proj, 0.001)
    @test τ < quantile(τ_proj, 0.999)

    # significance test
    pval_invvar_calib = GeoRDD.pval_invvar_calib(
        resid_GP_dict[T],
        resid_GP_dict[C],
        sentinels
    )
    @test pval_invvar_calib < 0.05 # effect should be detectable
end

@testset "run simulation" begin
    run_simulation()
end

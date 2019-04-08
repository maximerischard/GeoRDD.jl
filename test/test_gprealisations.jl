using Calculus
using DataFrames
using StatsModels
using LibGEOS
using GaussianProcesses
using GaussianProcesses: update_mll_and_dmll!, update_mll!, get_params, set_params!
using Distributions
using Random
using Optim
using LinearAlgebra

@testset "GPRealisations gradients" begin
    Random.seed!(1) # for replicability
    τ = 0.3 # constant treatment effect
    nobs = 100
    geordd_df, border_X1, border_X2, border_geo = sim_data(nobs, τ)

    fmla = @formula(outcome ~ GP(X1, X2) | region + covarA + covarB + categ)
    geordd = GeoRDD.regions_from_dataframe(fmla, geordd_df)
    # choose a Gaussian process kernel from the GaussianProcesses.jl package
    k_se = SEIso(log(0.5), log(1.0)) # Squared Exponential spatial kernel
    k_m = Const(log(20.0))           # constant kernel for the mean parameters
    βkern = LinIso(log(1.0))         # linear kernel for the other covariates
    logNoise = 1.0
    # create the Gaussian process object from the regional data:
    mgpcv = GeoRDD.MultiGPCovars(geordd, k_se+k_m, βkern, logNoise)
    param_kwargs = Dict(:domean=>false, :kern=>true, :noise=>true, :beta=>true)
    buf = Matrix{Float64}(undef, nobs, nobs)
    update_mll_and_dmll!(mgpcv, buf; param_kwargs...)
    grad = copy(mgpcv.dmll)
    init_x = get_params(mgpcv; param_kwargs...)

    numer_grad = Calculus.gradient(init_x) do params
        set_params!(mgpcv, params; param_kwargs...)
        update_mll!(mgpcv)
        t = mgpcv.mll
        t
    end
    @test grad ≈ numer_grad atol=1e-3
end

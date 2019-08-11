using GeoRDD
using Test

function sim_data(n::Int, τ::Real)
    X = rand(2, n)
    f_X = sin.(X[1,:] * 2.0) .+ 3.0 .* X[2,:].^2.0
    m = -0.8 # mean offset
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
    return geordd_df, border_X1, border_X2, border_geo
end
include("test_geometry.jl")
include("test_gprealisations.jl")
include("test_simulated.jl")


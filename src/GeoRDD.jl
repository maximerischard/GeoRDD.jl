module GeoRDD
    using Statistics
    using GaussianProcesses
    using Optim
    using LinearAlgebra
    import LibGEOS
    import GeoInterface
    import Combinatorics
    using Distributions: Normal, MultivariateNormal, ccdf, cdf
    using PDMats: AbstractPDMat
    import StatsModels
    using StatsModels: FormulaTerm
    import DataFrames
    using DataFrames: DataFrame, AbstractDataFrame
    import NLopt

    # utilities
    include("geometry.jl")
    # Types for multiple realizations of GPs
    include("regiondata.jl")
    module GPrealisations
        include("GPrealisations.jl")
        include("GPrealisationsCovars.jl")
        include("gp_utils.jl")
    end
    import .GPrealisations: GPRealisations, MultiGPCovars
    using .GPrealisations: postmean_β

    include("region_gp_fit.jl")
    # Cliff Face (treatment effect estimation)
    include("cliff_face.jl")
    # Local average treatment effect estimation
    include("border_projection.jl")
    include("average_treatment_effect.jl")
    include("weight_at_units.jl")
    # Hypothesis testing
    include("hypotest.jl")
    include("hypotest_chi2.jl")
    include("hypotest_invvar.jl")
    include("hypotest_mll.jl")
    include("placebo_geometry.jl")
    include("power.jl")
end

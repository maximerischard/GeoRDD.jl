module GeoRDD
    using Statistics
    import Statistics: mean
    using GaussianProcesses
    using PDMats
    using Optim
    import GeoInterface
    using GeoInterface: coordinates, xcoord, ycoord
    using LibGEOS: nearestPoints, interpolate, distance
    using LibGEOS: MultiPolygon, envelope
    import LibGEOS
    import LibGEOS: interpolate
    import Combinatorics
    using LinearAlgebra
    import Base: getindex, keys, values, start, done, next, iterate, eltype, length, size, convert
    using Distributions: Normal, MultivariateNormal, ccdf, cdf
    using PDMats: AbstractPDMat
    import GaussianProcesses: update_mll!, update_mll_and_dmll!, 
                              get_params, set_params!, num_params,
                              optimize!, GPE
    using GaussianProcesses: grad_stack, grad_stack!, grad_slice!, get_ααinvcKI!,
                             Mean, Kernel, KernelData, LinIso, MeanZero,
                             cov!, cov, cov_ij, dmll_kern!,
                             mat, cholfactors, wrap_cK, make_posdef!
    import StatsModels
    using StatsModels: Formula
    import DataFrames
    using DataFrames: DataFrame, AbstractDataFrame
    import NLopt

    # utilities
    include("gp_utils.jl")
    include("geometry.jl")
    # Types for multiple realizations of GPs
    include("regiondata.jl")
    include("GPrealisations.jl")
    include("GPrealisationsCovars.jl")
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

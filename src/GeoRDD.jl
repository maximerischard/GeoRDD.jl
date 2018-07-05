module GeoRDD
    using GaussianProcesses
    using PDMats
    using Optim
    import GeoInterface
    import LibGEOS
    import LibGEOS: interpolate
    import Combinatorics
    import Base: mean, getindex, keys, values, start, done, next, iteratorsize, iteratoreltype, eltype, length, size, convert
    using Distributions: Normal, MultivariateNormal, ccdf, cdf
    using PDMats: AbstractPDMat
    import GaussianProcesses: update_mll!, update_mll_and_dmll!, 
                              get_params, set_params!, num_params,
                              optimize!, GPE
    using GaussianProcesses: MatF64, VecF64,
                             grad_stack, grad_stack!, grad_slice!, get_ααinvcKI!,
                             Mean, Kernel, KernelData, LinIso,
                             cov!, cov, cov_ij, dmll_kern!
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

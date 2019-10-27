module GeoRDD
    using Statistics
    using GaussianProcesses
    using GaussianProcesses: predict_full, KernelData
    using Optim
    using LinearAlgebra
    import LibGEOS
    import GeoInterface
    import Combinatorics
    using Distributions: Normal, MultivariateNormal, ccdf, cdf
    using PDMats
    using PDMats: AbstractPDMat
    import StatsModels
    using StatsModels: FormulaTerm
    import DataFrames
    using DataFrames: DataFrame, AbstractDataFrame
    import NLopt
    import Base: iterate

    # utilities
    include("geometry.jl")
    # Types for multiple realizations of GPs
    include("regiondata.jl")
    include("gp_utils.jl")
    module GPrealisations
        using Statistics
        import GaussianProcesses: update_mll!, update_mll_and_dmll!, 
                                  get_params, set_params!, num_params,
                                  optimize!, GPE
        using GaussianProcesses: grad_stack, grad_stack!, grad_slice!, get_ααinvcKI!,
                                 Mean, Kernel, KernelData, LinIso, MeanZero,
                                 cov!, cov, cov_ij, dmll_kern!,
                                 mat, cholfactors, wrap_cK, make_posdef!,
                                 CovarianceStrategy, Scalar, FullCovariance,
                                 FullCovariancePrecompute
        using PDMats
        using LinearAlgebra
        using GaussianProcesses
        import Optim
        include("gp_utils.jl")
        include("GPrealisations.jl")
        include("GPrealisationsCovars.jl")
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

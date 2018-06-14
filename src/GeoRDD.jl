module GeoRDD
    using GaussianProcesses
    using PDMats
    using Optim
    import GeoInterface
    import LibGEOS
    # utilities
    include("gp_utils.jl")
    include("geometry.jl")
    # Types for multiple realizations of GPs
    include("GPrealisations.jl")
    include("GPrealisationsCovars.jl")
    # Cliff Face (treatment effect estimation)
    include("cliff_face.jl")
    # Local average treatment effect estimation
    include("border_projection.jl")
    include("average_treatment_effect.jl")
    include("weight_at_units.jl")
    # Hypothesis testing
    include("hypotest_chi2.jl")
    include("hypotest_invvar.jl")
    include("hypotest_mll.jl")
    include("placebo_geometry.jl")
end

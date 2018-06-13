module GeoRDD
    using GaussianProcesses
    using PDMats
    using Optim
    include("geometry.jl")
    include("gp_utils.jl")
    include("constant_kernel.jl")
    include("multigp_covars.jl")
    include("GPrealisations.jl")
    include("cliff_face.jl")
    include("point_estimates.jl")
    include("placebo_geometry.jl")
    include("boot_chi2test.jl")
    include("boot_mLLtest.jl")
    include("boot_invvar_test.jl")
    include("border_projection.jl")
end

function sim_power!(gpT::GPE, gpC::GPE, gpNull::GPE, τ::Float64, 
            treat::BitVector, Xb::AbstractMatrix,
                chi_null::Vector{Float64}, 
                mll_null::Vector{Float64}, 
                pval_invvar_null::Vector{Float64},
                Σcliff::PDMat,
                cK_T::AbstractMatrix,
                cK_C::AbstractMatrix,
                KCT::AbstractMatrix
               )
    # simulate data
    Ysim = prior_rand(gpNull)
    Ysim[treat] .+= τ

    # Modify data in GP objects
    gpT.y[:] = Ysim[treat]
    gpC.y[:] = Ysim[.!treat]
    gpNull.y[:] = Ysim

    # Update GPs
    update_alpha!(gpT)
    update_alpha!(gpC)
    update_alpha!(gpNull)

    gpT.mll = mll(gpT)
    gpC.mll = mll(gpC)
    gpNull.mll = mll(gpNull)
    
    # mll
    Δmll = gpT.mll + gpC.mll - gpNull.mll
    pval_mll = mean(mll_null .> Δmll)
    
    # χ2
    χ2 = chistat(gpT, gpC, Xb, Σcliff, cK_T, cK_C)
    pval_χ2 = mean(chi_null .> χ2)
    
    # inverse-var
    pval_invvar_obs = pval_invvar_uncalib(gpT, gpC, Xb, Σcliff, cK_T, cK_C)

    invvar_bootcalib = mean(pval_invvar_null .< pval_invvar_obs)
    invvar_calib = pval_invvar_calib(gpT, gpC, Xb, Σcliff, cK_T, cK_C, KCT)

    return (pval_mll,pval_χ2,pval_invvar_obs,invvar_bootcalib,invvar_calib)
end

function nsim_power(gpT::GPE, gpC::GPE, τ::Float64, 
                Xb::AbstractMatrix,
                chi_null::Vector{Float64}, 
                mll_null::Vector{Float64}, 
                pval_invvar_null::Vector{Float64},
                nsim::Int
               )
    gpT_mod = modifiable(gpT)
    gpC_mod = modifiable(gpC)
    gpNull = make_null(gpT_mod, gpC_mod)

    _, Σcliff = cliff_face(gpT, gpC, Xb)
    Σraw, chol = make_posdef!(copy(Σcliff))
    PDcliff = PDMat(Σraw, chol)
    cK_T = cov(gpT.kernel, Xb, gpT.x)
    cK_C = cov(gpC.kernel, Xb, gpC.x)
    KCT = cov(gpC.kernel, gpC.x, gpT.x)

    treat = BitVector(undef, gpNull.nobs)
    treat[:] .= false
    treat[1:gpT.nobs] .= true
    power_sims = [sim_power!(gpT_mod, gpC_mod, gpNull, τ, treat, Xb, 
                            chi_null, mll_null, pval_invvar_null,
                            PDcliff, cK_T, cK_C, KCT
                           )
                  for _ in 1:nsim];
    return power_sims
end

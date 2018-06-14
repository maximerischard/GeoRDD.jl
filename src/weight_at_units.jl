"""
    Compute unit weights.

    Inputs:
        gp: A fitted Gaussian process
        sentinels: a 2×R matrix of border sentinels
        weights: an R-vector of border weights

    Output:
        w_unit: A vector of weights for each unit in `gp`
"""
function weight_at_units(gp::GPE, sentinels::Matrix, weights::Vector)
    @assert size(sentinels, 2) == length(weights)
    K_Xb = cov(gp.k, gp.X, sentinels)
    K_XX = gp.cK
    w_unit = (K_XX \ K_Xb) * weights ./ sum(weights)
    return w_unit
end

"""
    Compute unit weights for the LATE estimator with uniform border weights.
"""
function weights_unif(gpT::GPE, gpC::GPE, nb::Int, border)
    Xb = sentinels(border, nb)
    w_border = ones(nb)
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

"""
    Compute unit weights for the LATE estimator with inverse-variance border weights.
"""
function weights_invvar(gpT::GPE, gpC::GPE, nb::Int, border)
    Xb = sentinels(border, nb)
    _, Σpost = cliff_face(gpT, gpC, Xb)
    w_border = Σpost \ ones(nb)
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

"""
    Compute unit weights for the projected finite-population LATE estimator.
"""
function weights_proj(gpT::GPE, gpC::GPE, border)
    Xb_treat = projection_points(gpT, border)
    Xb_ctrol = projection_points(gpC, border)
    Xb_projected = [Xb_treat Xb_ctrol]
    Xb = Xb_projected
    nb = size(Xb, 2)
    w_border = ones(nb)
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

"""
    Compute unit weights for the projected land LATE estimator.
"""
function weights_geo(gpT::GPE, gpC::GPE, nb::Int, border, region)
    nb = 100
    Xb_projected, w_border = infinite_proj_sentinels(
            gpT, gpC, border, region, 0.2, 2.0/nb)
    Xb = Xb_projected
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

"""
    Compute unit weights for the projected superpopulation LATE estimator.
"""
function weights_pop(gpT::GPE, gpC::GPE, density::Function, nb::Int, border, region)
    nb = 100
    Xb_projected, w_border = infinite_proj_sentinels(
            gpT, gpC, border, region, 0.2, 2.0/nb; density=density)
    Xb = Xb_projected
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

"""
    Compute unit weights for the population density-weighted LATE estimator.
"""
function weights_rho(gpT::GPE, gpC::GPE, density::Function, nb::Int, border)
    Xb = sentinels(border, nb)
    
    w_border = density.(Xb[1,:], Xb[2,:])
    w_units = [weight_at_units(gpT, Xb, w_border); -weight_at_units(gpC, Xb, w_border)]
    return Xb, w_border, w_units
end

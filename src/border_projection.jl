import GeoInterface: coordinates, xcoord, ycoord
import LibGEOS
import IterTools
import DataStructures
using LibGEOS: nearestPoints, interpolate, distance
using LibGEOS: MultiPolygon, envelope
import GaussianProcesses: GPE

function projection_points(gp::GPE, border::B; maxdist::Float64=Inf) where {B<:BorderType}
    X∂ = Array{Float64}(2, gp.nobsv)
    distances = Vector{Float64}(gp.nobsv)
    for i in 1:gp.nobsv
        # obtain coordinates for treatment point
        x,y = gp.X[:,i]
        point = LibGEOS.Point(x,y)
        # projection onto border (as distance along border)
        distances[i] = LibGEOS.distance(border, point)
        proj_point = nearestPoints(border, point)[1]
        # get border point from distance
        # proj_point = interpolate(border, proj_dist)
        # get coordinates from point
        proj_x, proj_y = coordinates(proj_point)
        X∂[1,i] = proj_x
        X∂[2,i] = proj_y
    end
    return X∂[:, distances .<= maxdist]
end

function proj_estimator(gpT::GPE, gpC::GPE, border::B; maxdist::Float64=Inf) where {B<:BorderType}
    X∂_treat = projection_points(gpT, border; maxdist=maxdist)
    X∂_ctrol = projection_points(gpC, border; maxdist=maxdist)
    X∂ = [X∂_treat X∂_ctrol]
    
    μpost, Σpost = cliff_face(gpT, gpC, X∂)
    return unweighted_mean(μpost, Σpost)
end

function data_hull(gpT::GPE, gpC::GPE)
    # Obtain a convex hull containing all the data.
    X_multi_treat = LibGEOS.MultiPoint([gpT.X[:, i] for i in 1:gpT.nobsv])
    X_multi_ctrol = LibGEOS.MultiPoint([gpC.X[:, i] for i in 1:gpC.nobsv])
    convexhull_treat = LibGEOS.convexhull(X_multi_treat)
    convexhull_ctrol = LibGEOS.convexhull(X_multi_ctrol)
    hull = LibGEOS.union(convexhull_treat, convexhull_ctrol)
    return hull
end

function infinite_proj_sentinels(gpT::GPE, gpC::GPE, border::B,
                                    region::MultiPolygon,
                                    maxdist::Float64, gridspace::Float64;
                                    density = ((s1,s2) -> 1.0)) where {B<:BorderType}
    # Add a buffer around the border, returns a "border area".
    #=buffer_polygon = LibGEOS.buffer(border, buffer)=#

    # Get the minimum and maximum of X1 and X2 within the region
    env = envelope(region)
    env_coord = coordinates(env)[1]
    X1_min = minimum(xcoord(p) for p in env_coord)
    X1_max = maximum(xcoord(p) for p in env_coord)
    X2_min = minimum(ycoord(p) for p in env_coord)
    X2_max = maximum(ycoord(p) for p in env_coord)

    # Obtain grid of points that covers the region
    X1_grid = X1_min:gridspace:X1_max
    X2_grid = X2_min:gridspace:X2_max

    border_envelope = envelope(border)

    projected_weights = Dict{Vector{Float64}, Float64}()
    for (s1,s2) in IterTools.product(X1_grid, X2_grid)
        # convert to a point obejct
        p = LibGEOS.Point(s1,s2)
        # Only keep points that are both within `maxdist` of
        # the border, and are in the convex hull of the data.
        if LibGEOS.distance(p, border_envelope) > maxdist
            continue
        elseif LibGEOS.distance(p, border) > maxdist
            continue
        elseif !LibGEOS.within(p, region)
            continue
        end
        # compute the population density at this location
        ρ = density(s1,s2)

        # project the point onto the border
        projected = LibGEOS.nearestPoints(p, border)[2]
        # obtain the coordinates of the projected point
        proj_coords = GeoInterface.coordinates(projected)[1:2]
        if haskey(projected_weights, proj_coords)
            projected_weights[proj_coords] += ρ
        else
            # initialize
            projected_weights[proj_coords] = ρ
        end
    end

    unique_projected_points = collect(keys(projected_weights))
    X∂_projected = [[p[1] for p in unique_projected_points]';
                    [p[2] for p in unique_projected_points]']
    # And the counts as the weights.
    weights = collect(values(projected_weights))
    # Note: assumes keys() and values() are in the same order
    return X∂_projected, weights
end

function infinite_proj_estim(gpT::GPE, gpC::GPE, border::B,
                                    region::MultiPolygon,
                                    maxdist::Float64, gridspace::Float64;
                                    density = ((s1,s2) -> 1.0)) where {B<:BorderType}
    X∂_projected, weights = infinite_proj_sentinels(
            gpT, gpC, border, region, maxdist, gridspace; density=density
            )
            
    # Obtain the cliff face for these sentinels...
    μpost, Σpost = cliff_face(gpT, gpC, X∂_projected)
    # ... and apply a weighted mean estimator.
    return weighted_mean(μpost, Σpost, weights)
end

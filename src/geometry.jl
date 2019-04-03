using LibGEOS: distance, envelope
import LibGEOS: interpolate
using Combinatorics: permutations, combinations
using GeoInterface: coordinates, xcoord, ycoord

# type conversion
const BorderType = Union{LibGEOS.MultiLineString, LibGEOS.LineString}
BorderType(ls::GeoInterface.LineString) = LibGEOS.LineString(ls)
BorderType(ls::GeoInterface.MultiLineString) = LibGEOS.MultiLineString(ls)

const RegionType = Union{LibGEOS.Polygon, LibGEOS.MultiPolygon}
convert(::Type{RegionType}, p::GeoInterface.AbstractPolygon) = LibGEOS.Polygon(p)
convert(::Type{RegionType}, p::GeoInterface.AbstractMultiPolygon) = LibGEOS.MultiPolygon(p)

function interpolate(border::LibGEOS.MultiLineString, dist::Real)
    ptr = border.ptr
    interp_ptr = LibGEOS.interpolate(ptr, dist)
    interp = LibGEOS.Point(interp_ptr)
    return interp
end
function sentinels(border::BorderType, nsent::Int)
    sentinel_distances = range(0, stop=LibGEOS.geomLength(border), length=nsent)
    sentinel_points = [interpolate(border,q) for q in sentinel_distances]
    sentinel_coords = [coordinates(p) for p in sentinel_points]
    sentinels_x = [p[1] for p in sentinel_coords]
    sentinels_y = [p[2] for p in sentinel_coords]
    Xb = [sentinels_x' ; sentinels_y']
    return Xb
end

function inregion(point::GeoInterface.Point, region::GeoInterface.MultiPolygon)
    xy = coordinates(point)
    poly_coords = coordinates(region)
    isin = false
    for poly in poly_coords
        if inpoly(xy, hcat(poly_coords)...)
            isin = true
        end
    end
    return isin
end
function raw_border(A_T::RegionType, A_C::RegionType, buffer::Float64)
    A_T_boundary = LibGEOS.boundary(A_T)
    A_C_buffered = LibGEOS.buffer(A_C, buffer)
    border = LibGEOS.intersection(A_T_boundary, A_C_buffered)
    return border
end
function get_border(A_T::RegionType, A_C::RegionType, buffer::Float64)
    if distance(envelope(A_T), envelope(A_C)) > buffer
        return nothing
    end
    if distance(A_T, A_C) > buffer
        return nothing
    end
    border = raw_border(A_T, A_C, buffer)
    if !(typeof(border) <: BorderType)
        return nothing
    end
    border = GeoRDD.rearrange_lines(border)
    return border
end
function get_border(A_T::P1, A_C::P2, buffer::Float64) where {
                P1 <: Union{GeoInterface.AbstractPolygon, GeoInterface.AbstractMultiPolygon}, 
                P2 <: Union{GeoInterface.AbstractPolygon, GeoInterface.AbstractMultiPolygon}}
    A_T_libgeos = convert(RegionType, A_T)
    A_C_libgeos = convert(RegionType, A_C)
    return get_border(A_T_libgeos, A_C_libgeos)
end

function region_grid(region::RegionType, gridspace::Float64)
    env = envelope(region)
    env_coord = coordinates(env)[1]
    X1_min = minimum(xcoord(p) for p in env_coord)
    X1_max = maximum(xcoord(p) for p in env_coord)
    X2_min = minimum(ycoord(p) for p in env_coord)
    X2_max = maximum(ycoord(p) for p in env_coord)

    # Obtain grid of points that covers the region
    X1_grid = X1_min:gridspace:X1_max
    X2_grid = X2_min:gridspace:X2_max
    gridpoints = LibGEOS.MultiPoint([
        [x, y]
        for x in X1_grid
        for y in X2_grid
       ])
    inregion_gridpoints = LibGEOS.intersection(region, gridpoints)
    grid_coords = coordinates(inregion_gridpoints)
    grid_mat = hcat(grid_coords...)
    return grid_mat
end
    

### What follows is tedious code to simplify a collection of linestrings
### into a multistring such that hops between adjacents linestrings
### are as short as possible
function merge_adjacent(segments)
    point_dict = Dict{Vector{Float64}, Vector{Vector{Float64}}}()
    start_dict = Dict{Vector{Float64}, Vector{Vector{Float64}}}()
    for seg in segments
        for new_point in (seg[1], seg[end])
            if haskey(point_dict, new_point)
                other_seg = point_dict[new_point]
                delete!(start_dict, other_seg[1])
                delete!(point_dict, other_seg[1])
                delete!(point_dict, other_seg[end])
                if seg[1] == other_seg[1]
                    # two segments have the same start
                    seg = vcat(reverse(seg), other_seg)
                elseif seg[end] == other_seg[1]
                    seg = vcat(seg, other_seg)
                elseif seg[1] == other_seg[end]
                    seg = vcat(other_seg, seg)
                elseif seg[end] == other_seg[end]
                    seg = vcat(other_seg, reverse(seg))
                else
                    throw("this makes no sense")
                end
            end
        end
        start_dict[seg[1]] = seg
        point_dict[seg[1]] = seg
        point_dict[seg[end]] = seg
    end
    return collect(values(start_dict))
end
""" Find all pairwise distances between segments.
    The distance between two segments is the minimum distances
    between their endpoints.
"""
function prep_distances(segments)
    nseg = length(segments)
    distances = Array{Float64}(undef, nseg, nseg)
    for j in 1:length(segments)
        for i in 1:length(segments)
            iseg = segments[i]
            jseg = segments[j]
            distances[i, j] = sqrt(
                min(
                    sum(x->x^2, iseg[1].-jseg[1]),
                    sum(x->x^2, iseg[1].-jseg[end]),
                    sum(x->x^2, iseg[end].-jseg[1]),
                    sum(x->x^2, iseg[end].-jseg[end])
                )
            )
        end
    end
    return distances
end
function travel_distance(distmat::Matrix{Float64}, order::AbstractVector{Int})
    nseg = length(order)
    sum(distmat[order[i], order[i+1]] for i in 1:nseg-1)
end
function _shortest_path_brute(distmat::Matrix{Float64})
    nseg = size(distmat, 1)
    mind = Inf
    shortest = collect(1:nseg)
    for order in permutations(1:nseg)
        d = travel_distance(distmat, order)
        if d < mind
            shortest = order
            mind = d
        end
    end
    return shortest
end
function _shortest_path_greedy(distmat::Matrix{Float64})
    nseg = size(distmat, 1)
    order = collect(1:nseg)
    mind = travel_distance(distmat, order)
    converged = false
    while !converged
        converged = true
        for swap_pair in combinations(1:nseg, 2)
            i, j = swap_pair
            order[i], order[j] = order[j], order[i]
            d = travel_distance(distmat, order)
            if d < mind
                mind = d
                converged = false
                break
            end
            # Put things back where they were if we didn't make progress.
            order[i], order[j] = order[j], order[i]
        end
    end
    return order
end
function shortest_path(distmat::Matrix{Float64})
    nseg = size(distmat, 1)
    if nseg <= 10
        return _shortest_path_brute(distmat)
    else
        return _shortest_path_greedy(distmat)
    end
end
function dist_orientation(segments, updown)
    nseg = length(segments)
    d = 0.0
    for i in 1:nseg-1
        if updown[i] # i is upright
            iend = segments[i][end]
        else # i us upside down
            iend = segments[i][1]
        end
        if updown[i+1]
            jstart = segments[i+1][1]
        else
            jstart = segments[i+1][end]
        end
        @assert length(iend) == 2
        @assert length(jstart) == 2
        d += sqrt(sum(x->x^2, iend.-jstart))
    end
    return d
end
function best_orientation(segments) 
    nseg = length(segments)
    shortest_updown = zeros(Bool, nseg)
    mindo = dist_orientation(segments, shortest_updown)
    for up in combinations(1:nseg)
        updown = zeros(Bool, nseg)
        updown[up] = true
        d = dist_orientation(segments, updown)
        if d < mindo
            shortest_updown = updown
            mindo = d
        end
    end
    return shortest_updown
end
function rearrange_lines(segments)
    merged = merge_adjacent(segments)
    if length(merged) == 1
        # nothing else to do
        return merged
    end
    distmat = prep_distances(merged)
    ordered = merged[shortest_path(distmat)]
    shortest_updown = best_orientation(ordered)
    nseg = length(ordered)
    for i in 1:nseg
        if !shortest_updown[i] # down
            ordered[i] = reverse(ordered[i])
        end
    end
    return ordered
end
function rearrange_lines(lines::T) where T <: GeoInterface.AbstractLineString
    # there's nothing to do if the border is just a single line
    return lines
end
function rearrange_lines(lines::T) where T <: GeoInterface.AbstractMultiLineString
    segments = coordinates(lines)
    ordered = rearrange_lines(segments)
    return T(ordered)
end


using LibGEOS: LineString, MultiLineString, geomLength, interpolate

# type conversion
const BorderType = Union{MultiLineString, LineString}
BorderType(ls::GeoInterface.LineString) = LineString(ls)
BorderType(ls::GeoInterface.MultiLineString) = MultiLineString(ls)

function sentinels(border::BorderType, nsent::Int)
    sentinel_distances = linspace(0,geomLength(border), nsent)
    sentinel_points = [interpolate(border,q) for q in sentinel_distances]
    sentinel_coords = [GeoInterface.coordinates(p) for p in sentinel_points]
    sentinels_x = [p[1] for p in sentinel_coords]
    sentinels_y = [p[2] for p in sentinel_coords]
    Xb = [sentinels_x' ; sentinels_y']
    return Xb
end

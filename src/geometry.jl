import LibGEOS: LineString, MultiLineString
import GeoInterface

# type conversion
const BorderType = Union{MultiLineString, LineString}
BorderType(ls::GeoInterface.LineString) = LineString(ls)
BorderType(ls::GeoInterface.MultiLineString) = MultiLineString(ls)

doc = """Geocoding NYC sales data by cross-referencing lots
with geospatial shapefiles.

Usage:
  NYC_dataprep.jl [--sales=<sales_xls>...] [--parcels=<parcels_shp>...] [--condos=<condos_dbf>]
"""
import CSV
using DataFrames
using Printf
import Dates
import Shapefile
import DBFTables
import ExcelFiles
import ExcelReaders
import Statistics
import Proj4
import GeoInterface
using GeoInterface: coordinates
import Proj4: transform
using DocOpt  # import docopt function

function main()
    salesfiles, parcelfiles, condosfile = parse_docopt()

    # concatenate dataframe 
    sales_df = vcat(read_sales_df.(salesfiles)...)
    # load parcel centroids and store information in dictionary
    sbl_to_taxinfo_map = Dict(union(sbl_to_taxinfo_pairs_from_shapefile.(parcelfiles)...))

    condo_bbl_map = DBFTables.Table(condosfile) |> map_condos_to_tax_lot
    sales_taxinfo = extract_sales_taxinfo(sales_df, sbl_to_taxinfo_map, condo_bbl_map)

    # add additional columns to sales dataframe
    sales_df[!,:XCoord] = [info.x for info in sales_taxinfo]
    sales_df[!,:YCoord] = [info.y for info in sales_taxinfo]
    sales_df[!,:SchDistr] = [info.schcode for info in sales_taxinfo]
    summary_of_missing(sales_df)

    CSV.write(Base.stdout, sales_df; dateformat="Y-mm-dd")
end

function parse_docopt()
    args = docopt(doc, version=v"2.0.0")

    salesfiles = args["--sales"]
    parcelfiles = args["--parcels"]
    condosfile = args["--condos"]

    print(stderr, "salesfiles: \n")
    show(stderr, salesfiles)
    print(stderr, "\n")
    print(stderr, "parcelfiles: \n")
    show(stderr, parcelfiles)
    print(stderr, "\n")
    print(stderr, "condosfile: \n")
    show(stderr, condosfile)
    print(stderr, "\n")
    return salesfiles, parcelfiles, condosfile
end


function summary_of_missing(sales_df)
    # proportion of lots that could not be geocoded
    print(stderr, "fraction of sales not geocoded: \n")
    show(stderr, Statistics.mean(ismissing.(sales_df[!,:XCoord])))
    print(stderr, "\n")
    # proportion of lots that could not be geocoded amongst 1- 2- 3-family dwellings
    dwellings = occursin.("DWELLINGS", sales_df.BUILDING_CLASS_CATEGORY)
    print(stderr, "fraction of dwellings not geocoded: \n")
    show(stderr, Statistics.mean(ismissing.(sales_df[dwellings,:XCoord])))
    print(stderr, "\n")
end

fix_alphanumeric_type(x::Real) = (x % 1.0 ≈ 0.0) ? string(Int(x)) : string(x)
fix_alphanumeric_type(x::String) = string(strip(x))
fix_alphanumeric_type(x) = string(x) # handle some dates in Apartment Number column
function read_sales_df(filepath)
    file = ExcelReaders.openxl(filepath)
    sheet1 = file.workbook.sheet_names()[1]
    # first 4 rows contain descriptions of the data
    sales_df = ExcelFiles.load(filepath, sheet1; skipstartrows=4) |> DataFrame
    # change spaces to underscores
    rename!(name -> CSV.normalizename(String(name)), sales_df)
    # clean up types
    for intkey in (:BOROUGH, :BLOCK, :LOT, :ZIP_CODE, :RESIDENTIAL_UNITS,
                   :COMMERCIAL_UNITS, :TOTAL_UNITS, :YEAR_BUILT)
        sales_df[!,intkey] = convert.(Int, sales_df[!,intkey])
    end
    for zeronullkey in (:LAND_SQUARE_FEET, :GROSS_SQUARE_FEET, :COMMERCIAL_UNITS, 
                        :TOTAL_UNITS, :RESIDENTIAL_UNITS, :SALE_PRICE, :YEAR_BUILT)
        sales_df[!,zeronullkey] = [x==0 ? missing : x for x in sales_df[!,zeronullkey]]
    end
    for alphanumkey in (:TAX_CLASS_AT_PRESENT, :TAX_CLASS_AT_TIME_OF_SALE, :APARTMENT_NUMBER)
        sales_df[!,alphanumkey] = fix_alphanumeric_type.(sales_df[!,alphanumkey])
    end

    for (key,dtype) in zip(names(sales_df), eltype.(eachcol(sales_df)))
        if dtype <: String
            sales_df[!,key] = String.(strip.(sales_df[!, key]))
        end
    end
    return sales_df
end

struct TaxInfo
    x::Union{Float64, Missing}
    y::Union{Float64, Missing}
    schcode::Union{String, Missing}
    address::Union{String, Missing}
end

function transform(src::Proj4.Projection, dest::Proj4.Projection, multicoords::AbstractVector{V} where V<:AbstractVector)
    return transform.(Ref(src), Ref(dest), multicoords)
end
Shapefile.Point(xy::Vector{Float64}) = Shapefile.Point(xy...)
function transform(src, dist, geom::G) where G<:GeoInterface.AbstractGeometry
    G(transform(src, dist, coordinates(geom)))
end
function transform(src, dist, p::Missing) missing end
function transform_epsg(coords; epsg_from::Int, epsg_to::Int)
    proj_from = Proj4.Projection(Proj4.epsg[epsg_from])
    proj_to = Proj4.Projection(Proj4.epsg[epsg_to])
    return transform(proj_from, proj_to, coords)
end

function sbl_to_taxinfo_pairs_from_shapefile(filepath)
    # read shapefile
    table = Shapefile.Table(filepath)
    # obtain the geometries (points)
    points = Shapefile.shapes(table)
    # school district code
    sch_code = table.SCH_CODE
    # and address
    address = [ismissing(st_nbr) ? missing : st_nbr*" "*street*", "*zipcode
               for (st_nbr, street, zipcode) 
               in zip(table.LOC_ST_NBR, table.LOC_STREET, table.LOC_ZIP)]
    projected_points = transform_epsg.(points; epsg_from=26918, epsg_to=2263)
    info = [TaxInfo(ismissing(p) ? missing : p.x,
                  ismissing(p) ? missing : p.y,
                  s, addr)
          for (p, s, addr)
          in zip(projected_points, sch_code, address)]
    # create a vector of pairs of SBL codes and points
    sbl_to_taxinfo_pairs = zip(table.SBL, info)
    return sbl_to_taxinfo_pairs
end

function map_condos_to_tax_lot(dtm_condo)
    condo_bbl_map = Dict{String,String}()
    for row in dtm_condo
        if ismissing(row.UNIT_BBL)
            continue
        elseif ismissing(row.CONDO_BA02)
            continue
        end
        condo_bbl_map[row.UNIT_BBL] = row.CONDO_BA02
    end
    return condo_bbl_map
end

function extract_sales_taxinfo(sales_df, sbl_to_taxinfo_map, condo_bbl_map)
    sales_sbl = sbl_from_components.(sales_df.BOROUGH, sales_df.BLOCK, sales_df.LOT)
    missing_info = TaxInfo(missing,missing,missing,missing)
    sales_taxinfo = TaxInfo[
        # if the SBL code is directly mapped to a TaxInfo record, use that
        if haskey(sbl_to_taxinfo_map, sbl)
            sbl_to_taxinfo_map[sbl]
        elseif haskey(condo_bbl_map, sbl)
            # otherwise it might be a condo, so
            # map the condo lot to a tax lot and
            # try again
            tax_sbl = condo_bbl_map[sbl]
            if haskey(sbl_to_taxinfo_map, tax_sbl)
                sbl_to_taxinfo_map[tax_sbl]
            else
                missing_info
            end
        else
            missing_info
        end
        for sbl in sales_sbl
    ]
    return sales_taxinfo
end

function sbl_from_components(borough::Int, block::Int, lot::Int)
    @sprintf("%01d%05d%04d", borough, block, lot)
end

main()

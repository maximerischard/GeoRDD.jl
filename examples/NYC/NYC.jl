import LibGEOS
import GeoJSON
import Base.convert
using DataFrames
using Distributions
import CSV
import JSON

import GeoInterface

const SchDistr = Int

const SALE_PRICE = Symbol("SALE PRICE")
const SQFT = Symbol("GROSS SQUARE FEET")
const BUILDING_CLASS_AT_TIME_OF_SALE = Symbol("BUILDING CLASS AT TIME OF SALE")
const BUILDING_CLASS_CATEGORY = Symbol("BUILDING CLASS CATEGORY")
const TAX_CLASS_AT_TIME_OF_SALE = Symbol("TAX CLASS AT TIME OF SALE")

function read_distr_shapes(; data_dir="NYC_data")
    json=GeoJSON.parsefile(joinpath(data_dir, "nysd_16c", "nysd.json"))
    schdistr_shape_dict = Dict{SchDistr, GeoRDD.RegionType}()
    for feature in json.features
        schdistr = convert(SchDistr, feature.properties["SchoolDist"])
        shape = convert(GeoRDD.RegionType, feature.geometry)
        schdistr_shape_dict[schdistr] = shape
    end
    return schdistr_shape_dict
end

function read_processed_sales(; data_dir="NYC_data")
    NYC_sales=CSV.read(joinpath(data_dir, "processed", "NYC_sales.csv"), 
                       DataFrame,
                       types=Dict("TAX CLASS AT PRESENT" => Union{Missings.Missing, String},
                                  "TAX CLASS AT TIME OF SALE" => Union{Missings.Missing, String}),
                       strings=:raw,
                       allowmissing=:all,
                      )
    nyc_schdistrs = CategoricalVector(NYC_sales[:SchDistr])
    # schd_strings = [ismissing(sd)?missing:dec(sd,2) for sd in  nyc_schdistrs]
    # str_schdistrs = CategoricalVector(schd_strings)
    NYC_sales[:SchDistr] = nyc_schdistrs
    # categorical variables
    categorical!(NYC_sales, Symbol[
        "BOROUGH",
        "BUILDING CLASS CATEGORY",
        "BUILDING CLASS AT TIME OF SALE",
        "ZIP CODE",
        "TAX CLASS AT PRESENT",
        "TAX CLASS AT TIME OF SALE",
        "NEIGHBORHOOD",
        ])
    sort!(NYC_sales, :SchDistr)
    return NYC_sales
end

const DWELLINGS_DICT = Dict(
    "01  ONE FAMILY DWELLINGS"=>true,
    "02  TWO FAMILY DWELLINGS"=>true,
    "03  THREE FAMILY DWELLINGS"=>true,
    "04  TAX CLASS 1 CONDOS"=>false,
    "05  TAX CLASS 1 VACANT LAND"=>false,
    "06  TAX CLASS 1 - OTHER"=>false,
    "07  RENTALS - WALKUP APARTMENTS"=>false,
    "08  RENTALS - ELEVATOR APARTMENTS"=>false,
    "09  COOPS - WALKUP APARTMENTS"=>false,
    "10  COOPS - ELEVATOR APARTMENTS"=>false,
    "11  SPECIAL CONDO BILLING LOTS"=>false,
    "11A CONDO-RENTALS"=>false,
    "12  CONDOS - WALKUP APARTMENTS"=>false, # why are these duplicated?
    "13  CONDOS - ELEVATOR APARTMENTS"=>false,
    "14  RENTALS - 4-10 UNIT"=>false,
    "15  CONDOS - 2-10 UNIT RESIDENTIAL"=>false,
    "16  CONDOS - 2-10 UNIT WITH COMMERCIAL UNIT"=>false,
    "17  CONDO COOPS"=>false,
    "18  TAX CLASS 3 - UNTILITY PROPERTIES"=>false,
    "21  OFFICE BUILDINGS"=>false,
    "22  STORE BUILDINGS"=>false,
    "23  LOFT BUILDINGS"=>false,
    "25  LUXURY HOTELS"=>false,
    "26  OTHER HOTELS"=>false,
    "27  FACTORIES"=>false,
    "28  COMMERCIAL CONDOS"=>false,
    "29  COMMERCIAL GARAGES"=>false,
    "30  WAREHOUSES"=>false,
    "31  COMMERCIAL VACANT LAND"=>false,
    "32  HOSPITAL AND HEALTH FACILITIES"=>false,
    "33  EDUCATIONAL FACILITIES"=>false,
    "34  THEATRES"=>false,
    "35  INDOOR PUBLIC AND CULTURAL FACILITIES"=>false,
    "36  OUTDOOR RECREATIONAL FACILITIES"=>false,
    "37  RELIGIOUS FACILITIES"=>false,
    "38  ASYLUMS AND HOMES"=>false,
    "41  TAX CLASS 4 - OTHER"=>false,
    "42  CONDO CULTURAL/MEDICAL/EDUCATIONAL/ETC"=>false,
    "43  CONDO OFFICE BUILDINGS"=>false,
    "44  CONDO PARKING"=>false,
    "45  CONDO HOTELS"=>false,
    "46  CONDO STORE BUILDINGS"=>false,
    "47  CONDO NON-BUSINESS STORAGE"=>false,
    "48  CONDO TERRACES/GARDENS/CABANAS"=>false,
    "49  CONDO WAREHOUSES/FACTORY/INDUS"=>false,
    )
const RESIDENTIAL_DICT = Dict(
    "01  ONE FAMILY DWELLINGS"=>true,
    "02  TWO FAMILY DWELLINGS"=>true,
    "03  THREE FAMILY DWELLINGS"=>true,
    "04  TAX CLASS 1 CONDOS"=>true,
    "05  TAX CLASS 1 VACANT LAND"=>false,
    "06  TAX CLASS 1 - OTHER"=>false,
    "07  RENTALS - WALKUP APARTMENTS"=>false,
    "08  RENTALS - ELEVATOR APARTMENTS"=>false,
    "09  COOPS - WALKUP APARTMENTS"=>true,
    "10  COOPS - ELEVATOR APARTMENTS"=>true,
    "11  SPECIAL CONDO BILLING LOTS"=>false,
    "11A CONDO-RENTALS"=>false,
    "12  CONDOS - WALKUP APARTMENTS"=>true, # why are these duplicated?
    "13  CONDOS - ELEVATOR APARTMENTS"=>true,
    "14  RENTALS - 4-10 UNIT"=>false,
    "15  CONDOS - 2-10 UNIT RESIDENTIAL"=>true,
    "16  CONDOS - 2-10 UNIT WITH COMMERCIAL UNIT"=>false,
    "17  CONDO COOPS"=>true,
    "18  TAX CLASS 3 - UNTILITY PROPERTIES"=>false,
    "21  OFFICE BUILDINGS"=>false,
    "22  STORE BUILDINGS"=>false,
    "23  LOFT BUILDINGS"=>false,
    "25  LUXURY HOTELS"=>false,
    "26  OTHER HOTELS"=>false,
    "27  FACTORIES"=>false,
    "28  COMMERCIAL CONDOS"=>false,
    "29  COMMERCIAL GARAGES"=>false,
    "30  WAREHOUSES"=>false,
    "31  COMMERCIAL VACANT LAND"=>false,
    "32  HOSPITAL AND HEALTH FACILITIES"=>false,
    "33  EDUCATIONAL FACILITIES"=>false,
    "34  THEATRES"=>false,
    "35  INDOOR PUBLIC AND CULTURAL FACILITIES"=>false,
    "36  OUTDOOR RECREATIONAL FACILITIES"=>false,
    "37  RELIGIOUS FACILITIES"=>false,
    "38  ASYLUMS AND HOMES"=>false,
    "41  TAX CLASS 4 - OTHER"=>false,
    "42  CONDO CULTURAL/MEDICAL/EDUCATIONAL/ETC"=>false,
    "43  CONDO OFFICE BUILDINGS"=>false,
    "44  CONDO PARKING"=>false,
    "45  CONDO HOTELS"=>false,
    "46  CONDO STORE BUILDINGS"=>false,
    "47  CONDO NON-BUSINESS STORAGE"=>false,
    "48  CONDO TERRACES/GARDENS/CABANAS"=>false,
    "49  CONDO WAREHOUSES/FACTORY/INDUS"=>false,
    )

function filter_sales(NYC_sales::DataFrame)

    logSalePricePerSQFT = log.(NYC_sales[SALE_PRICE]) .- log.(NYC_sales[SQFT])
    believable = zeros(Bool, size(NYC_sales,1))
    removed_reason = Dict(
        :residential => 0,
        :noprice => 0,
        :nosqft => 0,
        :geocode => 0,
        :toosmall => 0,
        :outlier => 0,
       )
    for i in 1:size(NYC_sales,1)
        if ismissing(NYC_sales[i,BUILDING_CLASS_AT_TIME_OF_SALE])
            # remove sales with missing covariates
            removed_reason[:residential] += 1
        elseif ismissing(NYC_sales[i,BUILDING_CLASS_CATEGORY])
            # remove sales with missing covariates
            removed_reason[:residential] += 1
        elseif !DWELLINGS_DICT[NYC_sales[i,BUILDING_CLASS_CATEGORY]]
            # remove non-residential properties
            # in fact, remove things that aren't dwellings (houses?)
            removed_reason[:residential] += 1
        elseif ismissing(NYC_sales[i,SALE_PRICE])
            # remove sales without a sale price
            removed_reason[:noprice] += 1
        elseif ismissing(NYC_sales[i,SQFT])
            # remove sales with missing GROSS SQUARE FEET information
            removed_reason[:nosqft] += 1
        elseif ismissing(NYC_sales[i,:XCoord])
            # remove properties with failed geocoding
            removed_reason[:geocode] += 1
        elseif ismissing(NYC_sales[i,:YCoord])
            # remove properties with failed geocoding
            removed_reason[:geocode] += 1
        elseif ismissing(NYC_sales[i,:SchDistr])
            # remove properties without a school district
            removed_reason[:geocode] += 1
        # elseif ismissing(NYC_sales[i,TAX_CLASS_AT_TIME_OF_SALE])
            # # remove sales with missing covariates
            # removed_reason[7] += 1
        elseif NYC_sales[i,SQFT] < 100.0
            # remove properties smaller to 100sqft (seems unlikely to be real)
            removed_reason[:toosmall] += 1
        elseif logSalePricePerSQFT[i] < 3.0
            # that's too cheap (remove outliers)
            removed_reason[:outlier] += 1
        elseif logSalePricePerSQFT[i] > 8.0
            # that's too expensive (remove outliers)
            removed_reason[:outlier] += 1
        else
            # otherwise keep
            believable[i] = true
        end
    end

    filtered = copy(NYC_sales[believable,:])
    filtered[:logSalePricePerSQFT] = convert.(Float64, logSalePricePerSQFT[believable])
    for col in (:XCoord, :YCoord)
        filtered[col] = convert.(Float64, filtered[col])
    end
    return Dict(
        :filtered => filtered,
        :believable => believable,
        :removed_reason => removed_reason,
        )
end

# function sales_dicts(NYC_sales::DataFrame)
    # schdistrs = sort(NYC_sales[:SchDistr].pool.levels)
    # schdistr_indices = Dict{SchDistr,Vector{Int}}()
    # schdistrs_col = NYC_sales[:SchDistr]
    # for distr in schdistrs
        # indices = find(schdistrs_col.refs .== find(schdistrs_col.pool.index .== distr)[1])
        # nobsv_schdistr = length(indices)
        # @printf("District %d has %d sales\n", distr, nobsv_schdistr)
        # schdistr_indices[distr] = indices
    # end
    # Y_dict=Dict{SchDistr, Vector{Float64}}()
    # X_dict=Dict{SchDistr, Array{Float64,2}}()
    # for distr in schdistrs
        # Y_dict[distr] = NYC_sales[schdistr_indices[distr], :logSalePricePerSQFT]
        # X_dict[distr] = NYC_sales[schdistr_indices[distr],[:XCoord, :YCoord]]
    # end
    # return Dict(
            # :schdistr_indices => schdistr_indices,
            # :schdistrs => schdistrs,
            # :X_dict => X_dict,
            # :Y_dict => Y_dict
            # )
# end

function table_τpair(τpost_pairs::Dict{Tuple{NYC.SchDistr,NYC.SchDistr},Normal}, include_distr::Vector{Int})
    τ_pair_nested = Dict{NYC.SchDistr, Dict{NYC.SchDistr, Normal}}()
    for (distrA, distrB) in keys(τpost_pairs)
        τpost = τpost_pairs[distrA, distrB]
        if distrA ∉ include_distr
            continue
        end
        if distrB ∉ include_distr
            continue
        end
        if distrA ∉ keys(τ_pair_nested)
            τ_pair_nested[distrA] = Dict{NYC.SchDistr, Normal}()
        end
        τ_pair_nested[distrA][distrB] = τpost
    end
    ;

    for distrA in sort(collect(keys(τ_pair_nested)))
        if isempty(τ_pair_nested[distrA])
            continue
        end
        @printf("\\( \\mathbf{%2d} \\)", distrA)
        for distrB in sort(collect(keys(τ_pair_nested[distrA])))
            τ = τ_pair_nested[distrA][distrB]
            @printf("& \\( \\mathbf{%2d:}~%+.2f \\pm %.2f \\)", distrB, -mean(τ), std(τ))
        end
        print("\\\\ \n")
    end
end

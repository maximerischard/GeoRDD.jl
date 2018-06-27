type RegionData
    X::Matrix{Float64} # dim×n spatial covariates
    y::Vector{Float64} # outcomes
    D::Matrix{Float64} # p×n non-spatial covariates
    shape::Nullable{RegionType}
end
function RegionData(X::MatF64, y::VecF64, D::MatF64)
    RegionData(X, y, D, Nullable{RegionType}())
end

residuals(rd::RegionData, β::VecF64) = rd.y - rd.D'β
residuals_data(rd::RegionData, β::VecF64) = RegionData(
        rd.X, 
        residuals(rd, β),
        Matrix{Float64}(0, length(rd.y)), # empty covariates
        rd.shape
       )

################
### Geometry ###
################

function get_border(rdA::RegionData, rdB::RegionData, args...)
    !isnull(rdA.shape) || throw("first region has no shape")
    !isnull(rdB.shape) || throw("second region has no shape")
    return get_border(get(rdA.shape), get(rdB.shape), args...)
end

#####################################
### Iterate over adjacent regions ###
#####################################

type AdjacentIterator{KEY}
    allpairs::Combinatorics.Combinations{Array{KEY,1}}
    rddict::Dict{KEY, RegionData}
    buffer::Float64
end

function start(adj::AdjacentIterator)
    itr = adj.allpairs
    combi = start(itr)
    while !done(itr, combi)
        pair, combi = next(itr, combi)
        di = adj.rddict[pair[1]]
        dj = adj.rddict[pair[2]]
        border = get_border(di, dj, adj.buffer)
        if border isa BorderType
            return (false, (pair, border), combi)
        end
    end
    return (true, )
end
function done(adj::AdjacentIterator, state)
    return state[1]
end
function next(adj::AdjacentIterator, state)
    itr = adj.allpairs
    _, prevpair, combi = state
    while !done(itr, combi)
        pair, combi = next(itr, combi)
        di = adj.rddict[pair[1]]
        dj = adj.rddict[pair[2]]
        border = get_border(di, dj, adj.buffer)
        if border isa BorderType
            return prevpair, (false, (pair, border), combi)
        end
    end
    return prevpair, (true, prevpair, combi)
end
iteratorsize(::AdjacentIterator) = Base.SizeUnknown()

function adjacent_pairs(rd_dict::Dict{KEY, RegionData}, buffer::Float64) where {KEY}
    allpairs = Combinatorics.Combinations(collect(keys(rd_dict)), 2)
    return AdjacentIterator{KEY}(allpairs, rd_dict, buffer)
end
    

##############################
### Parsing GeoRDD Formula ###
##############################

struct GeoRDDFormula
    outcome::Symbol
    spatial_covariates::Vector{Symbol}
    groupindic::Symbol
    lmformula::Formula
end
"""
    This (admittedly hacky) function parses a formula intended to specify a GeoRDD.
    The syntax for the formula is
        Y ~ GP(X1, X2) | Z + COVAR1 + COVAR2 + …
    where Y is the outcome variable, X1 and X2 are the two spatial covariates,
    Z is the treatment or group indicator, and COVAR1 and COVAR2 are additional
    non-spatial covariates.
"""
function parse_geordd_formula(fmla::Formula)
    # We're now going to go on a fishing expedition inside of this formula
    # to pick out the things we need for a GeoRDD.
    local gp_args # what are the spatial coodinate column names?
    local gp_term # what is the bit of the formula that corresponds to the GP?
    local groupindic # what is the group (e.g. treatment/control) indicator column?
    terms = StatsModels.Terms(fmla)
    for t in terms.terms
        evalterms = StatsModels.evt(t)
        for (i,evterm) in enumerate(evalterms)
            if StatsModels.is_call(evterm, :GP)
                # this is the Gaussian Process
                @assert evterm.args[1] == :GP # sanity check
                # inside the GP(., .) function, pick out the column name
                # of the spatial coordinates
                gp_args = Symbol.(evterm.args[2:end])
                gp_term = t
                if length(evalterms) != 2
                    throw("The spatial GP should be interacted with a group indicator.")
                end
                if t.args[1] != :|
                    throw("The group indicator should be separated from the GP by |.")
                end
                groupindic = [evalterms[j] for j in 1:length(evalterms) if i!=j][1]
            end
        end
    end
    # Remove the GP(X1, X2)|indic from the formula so we're just left with a
    # linear model formula:
    if !isa(fmla.lhs, Symbol)
        throw("The left hand side should specify the outcome column name")
    end
    Y = fmla.lhs
    lmformula = StatsModels.dropterm(fmla, gp_term)
    return GeoRDDFormula(Y, gp_args, groupindic, lmformula)
end

################################################
### Extracting region data from a DataFrames ###
################################################

function regions_from_dataframe(df::AbstractDataFrame, outcome::Symbol, groupindic::Symbol, spatial_covariates::Vector{Symbol}, lmformula::Formula)
    contrasts = Dict{Symbol,StatsModels.AbstractContrasts}()
    for col in StatsModels.Terms(lmformula).eterms
        contrasts[col] = StatsModels.FullDummyCoding()
    end
    covars_mf = StatsModels.ModelFrame(lmformula, df, contrasts=contrasts)
    mm = StatsModels.ModelMatrix(covars_mf)
    p = size(mm,2)
    D = mm.m' # p×n matrix of covariates
    spatialdim = length(spatial_covariates)

    group_col = df[groupindic]
    groupKeys = DataFrames.levels(group_col)
    KEY = eltype(groupKeys)
    regionDict = Dict{KEY, RegionData}()
    for key in groupKeys
        irows = find(group_col .== key)
        nobsv = length(irows)
        if nobsv == 0
            continue
        end
        X_key = Matrix{Float64}(spatialdim, nobsv)
        for p in 1:spatialdim
            X_key[p, :] = df[irows, spatial_covariates[p]]
        end
        Y_key = df[irows, outcome]
        D_key = D[:, irows]
        rd = RegionData(X_key, Y_key, D_key)
        regionDict[key] = rd
    end

    return regionDict
end

function regions_from_dataframe(df::AbstractDataFrame, fmla::Formula)
    parsed = parse_geordd_formula(fmla)
    regions_from_dataframe(df, parsed.outcome, parsed.groupindic, 
                          parsed.spatial_covariates, parsed.lmformula)
end

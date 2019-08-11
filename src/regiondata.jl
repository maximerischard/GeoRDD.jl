mutable struct RegionData
    x::Matrix{Float64} # dim×n spatial covariates
    y::Vector{Float64} # outcomes
    D::Matrix{Float64} # p×n non-spatial covariates
    shape::Union{RegionType,Nothing}
end
function RegionData(x::AbstractMatrix, y::AbstractVector, D::AbstractMatrix)
    RegionData(x, y, D, nothing)
end

residuals(rd::RegionData, β::AbstractVector) = rd.y - rd.D'β
residuals_data(rd::RegionData, β::AbstractVector) = RegionData(
        rd.x, 
        residuals(rd, β),
        Matrix{Float64}(undef, 0, length(rd.y)), # empty covariates
        rd.shape
       )

################
### Geometry ###
################

function get_border(rdA::RegionData, rdB::RegionData, args...)
    rdA.shape != nothing || throw("first region has no shape")
    rdB.shape != nothing || throw("second region has no shape")
    return get_border(rdA.shape, rdB.shape, args...)
end

#####################################
### Iterate over adjacent regions ###
#####################################

mutable struct AdjacentIterator{KEY}
    allpairs::Combinatorics.Combinations{Array{KEY,1}}
    rddict::Dict{KEY, RegionData}
    buffer::Float64
end

function next_combi_state(adj, combi_state)
    itr = adj.allpairs
    while combi_state !== nothing
        (pair, state) = combi_state
        di = adj.rddict[pair[1]]
        dj = adj.rddict[pair[2]]
        border = get_border(di, dj, adj.buffer)
        if border isa BorderType
            return ((pair, border), state)
        end
        combi_state = iterate(itr, state)
    end
    return nothing
end
function iterate(adj::AdjacentIterator)
    itr = adj.allpairs
    combi_state = iterate(itr)
    return next_combi_state(adj, combi_state)
end
function iterate(adj::AdjacentIterator, state)
    itr = adj.allpairs
    combi_state = iterate(itr, state)
    return next_combi_state(adj, combi_state)
end

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
    lmformula::FormulaTerm
end
"""
    This (admittedly hacky) function parses a formula intended to specify a GeoRDD.
    The syntax for the formula is
        Y ~ GP(X1, X2) | Z + COVAR1 + COVAR2 + …
    where Y is the outcome variable, X1 and X2 are the two spatial covariates,
    Z is the treatment or group indicator, and COVAR1 and COVAR2 are additional
    non-spatial covariates.
"""
function parse_geordd_formula(fmla::FormulaTerm)
    # We're now going to go on a fishing expedition inside of this formula
    # to pick out the things we need for a GeoRDD.
    local gp_args # what are the spatial coodinate column names?
    local gp_term # what is the bit of the formula that corresponds to the GP?
    local groupindic # what is the group (e.g. treatment/control) indicator column?
    terms = StatsModels.terms(fmla)
    found = false
    for t in fmla.rhs # iterate over the RHS of the formula
        if t isa StatsModels.FunctionTerm
            left = t.args_parsed[1]
            if (left isa StatsModels.FunctionTerm) & (nameof(left.forig) == :GP)
                # this is the Gaussian process
                # inside the GP(., .) function, pick out the column name
                gp_args = Symbol.(left.args_parsed)
                gp_term = t
                if length(t.args_parsed) != 2
                    throw("The spatial GP should be interacted with a group indicator.")
                end
                if nameof(t.forig) != :|
                    throw("The group indicator should be separated from the GP by |.")
                end
                groupindic = Symbol(t.args_parsed[2])
                found = true
            end
        end
    end
    if !found
        throw("""Error: GP term not found in formula.
        
            The syntax for the formula is
                Y ~ GP(X1, X2) | Z + COVAR1 + COVAR2 + …
            where Y is the outcome variable, X1 and X2 are the two spatial covariates,
            Z is the treatment or group indicator, and COVAR1 and COVAR2 are additional
            non-spatial covariates.
            """)
    end
    Y = Symbol(fmla.lhs)
    # Remove the GP(X1, X2)|indic from the formula so we're just left with a
    # linear model formula:
    lmformula = StatsModels.drop_term(fmla, gp_term)
    return GeoRDDFormula(Y, gp_args, groupindic, lmformula)
end

################################################
### Extracting region data from a DataFrames ###
################################################

function regions_from_dataframe(df::AbstractDataFrame, outcome::Symbol, groupindic::Symbol, spatial_covariates::Vector{Symbol}, lmformula::FormulaTerm)
    contrasts = Dict{Symbol,StatsModels.AbstractContrasts}()
    for col in Symbol.(StatsModels.terms(lmformula))
        contrasts[col] = StatsModels.FullDummyCoding()
    end
    covars_mf = StatsModels.ModelFrame(lmformula, df, contrasts=contrasts)
    mm = StatsModels.ModelMatrix(covars_mf)
    p = size(mm,2)
    D = mm.m' # p×n matrix of covariates
    spatialdim = length(spatial_covariates)

    group_col = df[!,groupindic]
    groupKeys = DataFrames.levels(group_col)
    KEY = eltype(groupKeys)
    regionDict = Dict{KEY, RegionData}()
    for key in groupKeys
        irows = findall(group_col .== key)
        nobs = length(irows)
        if nobs == 0
            continue
        end
        X_key = Matrix{Float64}(undef, spatialdim, nobs)
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

function regions_from_dataframe(fmla::FormulaTerm, df::AbstractDataFrame)
    parsed = parse_geordd_formula(fmla)
    regions_from_dataframe(df, parsed.outcome, parsed.groupindic, 
                          parsed.spatial_covariates, parsed.lmformula)
end

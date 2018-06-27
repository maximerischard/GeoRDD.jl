"""
    Fit regional data using Gaussian processes.
"""

GPE(rd::RegionData, args...) = GPE(rd.X, rd.y, args...)

function MultiGPCovars(rdict::Dict{KEY,RegionData}, groupKeys::AbstractVector{KEY}, spkern::Kernel, βkern::Kernel, σ::Float64) where {KEY}
    Dlist = Matrix{Float64}[]
    gpList = GPE[]
    for key in groupKeys
        rd = rdict[key]
        D = rd.D
        y = rd.y
        m = MeanConst(mean(y))
        gp = GPE(rd.X, rd.y, m, spkern, log(σ))
        push!(gpList, gp)
        push!(Dlist, D)
    end
    return MultiGPCovars(Dlist, gpList, groupKeys, βkern)
end
function MultiGPCovars(rdict::Dict{KEY,RegionData}, spkern::Kernel, βkern::Kernel, σ::Float64) where {KEY}
    groupKeys = collect(keys(rdict))
    return MultiGPCovars(rdict, groupKeys, spkern, βkern, σ)
end

function GPRealisations(rdict::Dict{KEY,RegionData}, groupKeys::AbstractVector{KEY}, spkern::Kernel, σ::Float64) where {KEY}
    gpList = GPE[]
    for key in groupKeys
        rd = rdict[key]
        y = rd.y
        m = MeanConst(mean(y))
        gp = GPE(rd.X, rd.y, m, spkern, log(σ))
        push!(gpList, gp)
    end
    return GPRealisations(gpList, groupKeys)
end

function GPRealisations(rdict::Dict{KEY,RegionData}, spkern::Kernel, σ::Float64) where KEY
    groupKeys = collect(keys(rdict))
    return GPRealisations(rdict, groupKeys, spkern, σ)
end

using GaussianProcesses: Mean, Kernel, GPE

GPE(rd::RegionData, args...) = GPE(rd.x, rd.y, args...)

function MultiGPCovars(rdict::Dict{KEY,RegionData}, groupKeys::AbstractVector{KEY}, spkern::Kernel, βkern::Kernel, σ::Float64, mean::Mean=MeanZero()) where {KEY}
    Dlist = Matrix{Float64}[]
    gpList = GPE[]
    for key in groupKeys
        rd = rdict[key]
        D = rd.D
        y = rd.y
        gp = GPE(rd.x, rd.y, mean, spkern, log(σ))
        push!(gpList, gp)
        push!(Dlist, D)
    end
    return MultiGPCovars(Dlist, gpList, groupKeys, βkern)
end
function MultiGPCovars(rdict::Dict{KEY,RegionData}, spkern::Kernel, args...) where {KEY}
    groupKeys = collect(keys(rdict))
    return MultiGPCovars(rdict, groupKeys, spkern, args...)
end

function GPRealisations(rdict::Dict{KEY,RegionData}, groupKeys::AbstractVector{KEY}, spkern::Kernel, σ::Float64, mean::Mean=MeanZero()) where {KEY}
    gpList = GPE[]
    for key in groupKeys
        rd = rdict[key]
        y = rd.y
        gp = GPE(rd.x, rd.y, mean, spkern, log(σ))
        push!(gpList, gp)
    end
    return GPRealisations(gpList, groupKeys)
end

function GPRealisations(rdict::Dict{KEY,RegionData}, spkern::Kernel, args...) where {KEY}
    groupKeys = collect(keys(rdict))
    return GPRealisations(rdict, groupKeys, spkern, args...)
end

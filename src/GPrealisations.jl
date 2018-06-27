type GPRealisations{KEY}
    groupKeys::Vector{KEY}
    mgp::Dict{KEY,GPE}
    nobsv::Int
    logNoise::Float64
    m::Mean
    k::Kernel
    mll::Float64            # Marginal log-likelihood
    dmll::Vector{Float64}   # Gradient marginal log-likelihood
    function GPRealisations{KEY}(
            groupKeys::Vector{KEY},
            mgp::Dict{KEY,GPE}, 
            nobsv::Int,
            logNoise::Float64,
            m::Mean,
            k::Kernel,
            ) where {KEY}
        length(groupKeys) == length(mgp) || throw("groupKeys and mgp should have same length")
        gpreals = new(groupKeys, mgp, nobsv, logNoise, m, k)
        propagate_params!(gpreals)
        update_mll!(gpreals)
        return gpreals
    end
end
function GPRealisations(gpList::Vector{GPE}, groupKeys::Vector{KEY}) where {KEY}
    total_nobsv = sum([gp.nobsv for gp in gpList])
    # get kernel and parameters from first GP in list
    first_gp = gpList[1]
    dim = first_gp.dim
    logNoise = first_gp.logNoise
    kern = first_gp.k
    m = first_gp.m
    ngroups = length(groupKeys)
    @assert ngroups == length(gpList)
    gpDict = Dict{KEY, GPE}()
    istart = 1
    for j in 1:ngroups
        gp = gpList[j]
        key = groupKeys[j]
        nobsv = gp.nobsv
        gp.nobsv > 0 || throw("empty group")
        gpDict[key] = gp
        istart += nobsv
    end
    gpreals = GPRealisations{KEY}(
                groupKeys, gpDict, 
                total_nobsv, logNoise, m, kern)
    return gpreals
end

getindex(gpreals::GPRealisations{KEY}, key::KEY) where {KEY} = gpreals.mgp[key]
keys(gpreals::GPRealisations) = gpreals.groupKeys
values(gpreals::GPRealisations) = values(gpreals.mgp)

function get_params(gpreals::GPRealisations; 
                    noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gpreals.logNoise); end
    if domean
        for gp in values(gpreals.mgp)
            append!(params, get_params(gp.m))
        end
    end
    if kern; append!(params, get_params(gpreals.k)); end
    return params
end
function propagate_params!(gpreals::GPRealisations; noise::Bool=true, kern::Bool=true)
    for gp in values(gpreals.mgp)
        # harmonize parameters
        if kern
            gp.k = gpreals.k
        end
        if noise
            gp.logNoise = gpreals.logNoise
        end
    end
end
function set_params!(gpreals::GPRealisations, hyp::Vector{Float64}; 
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    istart=1
    if noise
        gpreals.logNoise = hyp[istart]
        istart += 1
    end
    if domean
        for key in gpreals.groupKeys
            gp = gpreals.mgp[key]
            set_params!(gp.m, hyp[istart:istart+num_params(gp.m)-1])
            istart += num_params(gp.m)
        end
    end
    if kern
        set_params!(gpreals.k, hyp[istart:end])
    end
    propagate_params!(gpreals, noise=noise, kern=kern)
end

function update_mll!(gpreals::GPRealisations)
    mll = 0.0
    for gp in values(gpreals.mgp)
        update_mll!(gp)
        mll += gp.mll
    end
    gpreals.mll = mll
    return mll
end
function update_mll_and_dmll!(gpreals::GPRealisations, 
                              ααinvcKIs::Dict{Int,Matrix}; 
                              noise::Bool=true, domean::Bool=true, kern::Bool=true)
    gpreals.mll = 0.0
    gpreals.dmll = zeros(get_params(gpreals; noise=noise, domean=domean, kern=kern))
    imean=2
    ikern=collect((length(gpreals.dmll)-num_params(gpreals.k)+1):length(gpreals.dmll))
    for gp in values(gpreals.mgp)
        update_mll_and_dmll!(gp, ααinvcKIs[gp.nobsv]; 
            noise=noise,  domean=domean, kern=kern)
        gpreals.mll += gp.mll
        dmll_indices=Int[]
        if noise
            push!(dmll_indices, 1)
        end
        if domean
            append!(dmll_indices, imean:imean+num_params(gp.m)-1)
            imean+=num_params(gp.m)
        end
        if kern
            append!(dmll_indices, ikern)
        end
        gpreals.dmll[dmll_indices] .+= gp.dmll
    end
    return gpreals.dmll
end
function update_mll_and_dmll!(gpreals::GPRealisations; kwargs...)
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in values(gpreals.mgp)
        if haskey(ααinvcKIs, gp.nobsv)
            continue
        end
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    return update_mll_and_dmll!(gpreals, ααinvcKIs)
end

function get_optim_target(gpreals::GPRealisations;
                          noise::Bool=true, domean::Bool=true, kern::Bool=true)
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in values(gpreals.mgp)
        if haskey(ααinvcKIs, gp.nobsv)
            continue
        end
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    function mll(hyp::Vector{Float64})
        try
            set_params!(gpreals, hyp; noise=noise, domean=domean, kern=kern)
            update_mll!(gpreals)
            return -gpreals.mll
        catch err
             if !all(isfinite(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end        
    end

    function mll_and_dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        set_params!(gpreals, hyp; noise=noise, domean=domean, kern=kern)
        update_mll_and_dmll!(gpreals, ααinvcKIs; noise=noise, domean=domean, kern=kern)
        grad[:] = -gpreals.dmll
        return -gpreals.mll
    end
    function dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        mll_and_dmll!(grad, hyp)
    end

    func = OnceDifferentiable(mll, dmll!, mll_and_dmll!,
        get_params(gpreals;noise=noise,domean=domean,kern=kern))
    return func
end
function optimize!(gpreals::GPRealisations;
                   noise::Bool=true, domean::Bool=true, kern::Bool=true)
    func = get_optim_target(gpreals, noise=noise, domean=domean, kern=kern)
    init = get_params(gpreals;  noise=noise, domean=domean, kern=kern)  # Initial hyperparameter values
    nparams = length(init)
    lower = -Inf*ones(nparams)
    upper = Inf*ones(nparams)
    ikernstart=1
    if noise
        ikernstart+=1
        lower[1]=-10.0
    end
    upper[ikernstart:ikernstart+num_params(gpreals.k)-1]=20.0
    best_x = copy(init)
    best_y = Inf
    count = 0
    function myfunc(x::Vector, grad::Vector)
        count += 1
        if length(grad) > 0
            y = func.fg!(grad, x)
            if y < best_y
                best_x[:] = x
            end
            return y
        else
            try
                y = func.f(x)
                if y < best_y
                    best_x[:] = x
                end
                return y
            catch
                return Inf
            end
        end
    end     

    opt = Opt(:LD_LBFGS, nparams)
    lower_bounds!(opt, lower)
    upper_bounds!(opt, upper)
    min_objective!(opt, myfunc)
    xtol_rel!(opt,1e-4)
    ftol_rel!(opt, 1e-20)

    (minf,minx,ret) = NLopt.optimize(opt, init)
    myfunc(best_x, [])
    return best_x, count
end

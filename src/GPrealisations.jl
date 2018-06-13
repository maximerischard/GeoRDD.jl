using Optim
using GaussianProcesses: Mean, Kernel, evaluate, metric
import GaussianProcesses: optimize!, get_optim_target
import GaussianProcesses: num_params, set_params!, get_params
import GaussianProcesses: update_mll_and_dmll!, update_mll!
using NLopt

type GPRealisations
    reals::Vector{GPE}
    k::Kernel
    logNoise::Float64
    mll::Float64
    dmll::Vector{Float64}
end

function GPRealisations(reals::Vector{GPE})
    first = reals[1]
    gpr = GPRealisations(reals, first.k, first.logNoise, NaN, [])
end

function get_params(gpr::GPRealisations; 
                    noise::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gpr.logNoise); end
    if mean
        for gp in gpr.reals
            append!(params, get_params(gp.m))
        end
    end
    if kern; append!(params, get_params(gpr.k)); end
    return params
end
function propagate_params!(gpr::GPRealisations; noise::Bool=true, kern::Bool=true)
    for gp in gpr.reals
        # harmonize parameters
        if kern
            gp.k = gpr.k
        end
        if noise
            gp.logNoise = gpr.logNoise
        end
    end
end
function set_params!(gpr::GPRealisations, hyp::Vector{Float64}; 
                     noise::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    istart=1
    if noise
        gpr.logNoise = hyp[istart]
        istart += 1
    end
    if mean
        for gp in gpr.reals
            set_params!(gp.m, hyp[istart:istart+num_params(gp.m)-1])
            istart += num_params(gp.m)
        end
    end
    if kern
        set_params!(gpr.k, hyp[istart:end])
    end
    propagate_params!(gpr, noise=noise, kern=kern)
end

function update_mll!(gpr::GPRealisations)
    mll = 0.0
    for gp in gpr.reals
        update_mll!(gp)
        mll += gp.mll
    end
    gpr.mll = mll
    return mll
end
function update_mll_and_dmll!(gpr::GPRealisations, 
                              Kgrads::Dict{Int,Matrix}, ααinvcKIs::Dict{Int,Matrix}; 
                              noise::Bool=true, mean::Bool=true, kern::Bool=true)
    gpr.mll = 0.0
    gpr.dmll = zeros(get_params(gpr; noise=noise, mean=mean, kern=kern))
    imean=2
    ikern=collect((length(gpr.dmll)-num_params(gpr.k)+1):length(gpr.dmll))
    for gp in gpr.reals
        update_mll_and_dmll!(gp, Kgrads[gp.nobsv], ααinvcKIs[gp.nobsv]; 
            noise=noise,mean=mean,kern=kern)
        gpr.mll += gp.mll
        dmll_indices=Int[]
        if noise
            push!(dmll_indices, 1)
        end
        if mean
            append!(dmll_indices, imean:imean+num_params(gp.m)-1)
            imean+=num_params(gp.m)
        end
        if kern
            append!(dmll_indices, ikern)
        end
        gpr.dmll[dmll_indices] .+= gp.dmll
    end
    return gpr.dmll
end
function update_mll_and_dmll!(gpr::GPRealisations; kwargs...)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    return update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs)
end

function get_optim_target(gpr::GPRealisations;
                          noise::Bool=true, mean::Bool=true, kern::Bool=true)
    Kgrads = Dict{Int,Matrix}()
    ααinvcKIs = Dict{Int,Matrix}()
    for gp in gpr.reals
        if haskey(Kgrads, gp.nobsv)
            continue
        end
        Kgrads[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
        ααinvcKIs[gp.nobsv] = Array{Float64}(gp.nobsv, gp.nobsv)
    end
    function mll(hyp::Vector{Float64})
        try
            set_params!(gpr, hyp; noise=noise, mean=mean, kern=kern)
            update_mll!(gpr)
            return -gpr.mll
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
        #=try=#
            set_params!(gpr, hyp; noise=noise, mean=mean, kern=kern)
            update_mll_and_dmll!(gpr, Kgrads, ααinvcKIs; noise=noise, mean=mean, kern=kern)
            grad[:] = -gpr.dmll
            return -gpr.mll
        #=catch err=#
        #=     if !all(isfinite(hyp))=#
        #=        println(err)=#
        #=        return Inf=#
        #=    elseif isa(err, ArgumentError)=#
        #=        println(err)=#
        #=        return Inf=#
        #=    elseif isa(err, Base.LinAlg.PosDefException)=#
        #=        println(err)=#
        #=        return Inf=#
        #=    else=#
        #=        throw(err)=#
        #=    end=#
        #=end =#
    end
    function dmll!(grad::Vector{Float64}, hyp::Vector{Float64})
        mll_and_dmll!(grad, hyp)
    end

    func = OnceDifferentiable(mll, dmll!, mll_and_dmll!,
        get_params(gpr;noise=noise,mean=mean,kern=kern))
    return func
end
#=function optimize!(gpr::GPRealisations; noise::Bool=true, mean::Bool=true, kern::Bool=true,=#
#=                    method=ConjugateGradient(), kwargs...)=#
#=    func = get_optim_target(gpr, noise=noise, mean=mean, kern=kern)=#
#=    init = get_params(gpr;  noise=noise, mean=mean, kern=kern)  # Initial hyperparameter values=#
#=    results=optimize(func,init; method=method, kwargs...)                     # Run optimizer=#
#=    set_params!(gpr, Optim.minimizer(results), noise=noise,mean=mean,kern=kern)=#
#=    update_mll!(gpr)=#
#=    return results=#
#=end=#
function optimize!(gpr::GPRealisations;
                   noise::Bool=true, mean::Bool=true, kern::Bool=true)
    func = get_optim_target(gpr, noise=noise, mean=mean, kern=kern)
    init = get_params(gpr;  noise=noise, mean=mean, kern=kern)  # Initial hyperparameter values
    nparams = length(init)
    lower = -Inf*ones(nparams)
    upper = Inf*ones(nparams)
    ikernstart=1
    if noise
        ikernstart+=1
        lower[1]=-10.0
    end
    upper[ikernstart:ikernstart+num_params(gpr.k)-1]=20.0
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

    # try
    (minf,minx,ret) = NLopt.optimize(opt, init)
    # catch
        # try
            # # LD_MMA seems to be slower
            # # but maybe more reliable
            # println("trying LD_MMA")
            # opt = Opt(:LD_MMA, nparams)
            # lower_bounds!(opt, lower)
            # upper_bounds!(opt, upper)
            # min_objective!(opt, myfunc)
            # xtol_rel!(opt, 1e-4)
            # ftol_rel!(opt, 1e-8)
            # (minf,minx,ret) = NLopt.optimize(opt, init)
        # catch
            # _ = myfunc(best_x, [])
            # return best_x, count
        # end
    # end
    _ = myfunc(best_x, [])
    return best_x, count
end

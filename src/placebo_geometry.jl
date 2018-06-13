immutable Point
    x::Float64
    y::Float64
end
function isLeft(a::Point, b::Point, c::Point)
     return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0
end
function left_points(angle::Float64, shift::Float64, X::Matrix{Float64})
    dydx=tand(angle)
    perp=tand(angle+90)
    meanx = mean(X[1,:])
    meany = mean(X[2,:])
    direction=sign(cosd(angle+90))
    if direction==0.0
        direction=1.0
    end
    shift_x = shift*cosd(angle+90)*direction
    shift_y = shift*sind(angle+90)*direction
    a = Point(meanx+shift_x,meany+shift_y)
    if abs(dydx)<1.0
        dx=1.0
        dy=dydx
    else
        dx=1.0/dydx
        dy=1.0
    end
    b = Point(meanx+shift_x+dx,meany+dy+shift_y)
    
    n = size(X,2)
    points = [Point(X[1,i],X[2,i]) for i in 1:n]
    are_left = [isLeft(a,b,p) for p in points]
end

function plot_line(angle::Float64, shift::Float64, X::Matrix; kwargs...)
    meanx=mean(X[1,:])
    meany=mean(X[2,:])
    dydx=tand(angle)
    direction=sign(cosd(angle+90))
    if direction==0.0
        direction=1.0
    end
    shift_x = shift*cosd(angle+90)*direction
    shift_y = shift*sind(angle+90)*direction
    
    xlim=plt.xlim()
    ylim=plt.ylim()
    xlim_arr = Float64[xlim[1],xlim[2]]
    ylim_arr = Float64[ylim[1],ylim[2]]
    if dydx > 1e3
        plt.axvline(meanx+shift_x, color="red"; kwargs...)
    elseif dydx > 10
        plt.plot(meanx+(ylim_arr.-meany.-shift_y)./dydx+shift_x,ylim_arr; color="red", kwargs...)
    else
        plt.plot(xlim_arr,meany+(xlim_arr.-shift_x.-meanx).*dydx+shift_y; color="red", kwargs...)
    end
    plt.xlim(xlim)
    plt.ylim(ylim)
end

function placebo_sentinels(angle::Float64, shift::Float64, X::MatF64, npoints::Int)
    meanx=mean(X[1,:])
    meany=mean(X[2,:])
    minx,maxx = extrema(X[1,:])
    miny,maxy = extrema(X[2,:])

    dydx=tand(angle)
    direction=sign(cosd(angle+90))
    if direction==0.0
        direction=1.0
    end
    shift_x = shift*cosd(angle+90)*direction
    shift_y = shift*sind(angle+90)*direction

    center_x = meanx + shift_x
    center_y = meany + shift_y

    y_from_minx = center_y + dydx*(minx-center_x)
    y_from_maxx = center_y + dydx*(maxx-center_x)
    x_from_miny = center_x + (miny-center_y)/dydx
    x_from_maxy = center_x + (maxy-center_y)/dydx

    extremes = []
    if miny < y_from_minx < maxy
        push!(extremes, (minx, y_from_minx))
    end
    if miny < y_from_maxx < maxy
        push!(extremes, (maxx, y_from_maxx))
    end
    if minx < x_from_miny < maxx
        push!(extremes, (x_from_miny, miny))
    end
    if minx < x_from_maxy < maxx
        push!(extremes, (x_from_maxy, maxy))
    end
    @assert length(extremes)==2
    x∂ = linspace(extremes[1][1], extremes[2][1], npoints)
    y∂ = linspace(extremes[1][2], extremes[2][2], npoints)
    X∂ = Float64[x∂'; y∂']
    return X∂
end

function prop_left(angle::Float64, shift::Float64, X::Matrix{Float64})
    are_left = left_points(angle, shift, X)
    return mean(are_left)
end

#copy-pasted from http://blog.mmast.net/bisection-method-julia
function bisection(f::Function, a::Number, b::Number;
                   tol::AbstractFloat=1e-5, maxiter::Integer=100)
    fa = f(a)
    fa*f(b) <= 0 || error("No real root in [a,b]")
    i = 0
    local c
    while b-a > tol
        i += 1
        i != maxiter || error("Max iteration exceeded")
        c = (a+b)/2
        fc = f(c)
        if fc == 0
            break
        elseif fa*fc > 0
            a = c  # Root is in the right half of [a,b].
            fa = fc
        else
            b = c  # Root is in the left half of [a,b].
        end
    end
    return c
end

function shift_for_even_split(angle::Float64, X::Matrix)
    minx,maxx = extrema(X[1,:])
    miny,maxy = extrema(X[2,:])
    maxdistance = sqrt( (maxy-miny)^2 + (maxx-minx)^2)
    return bisection(shift -> prop_left(angle,shift,X)-0.5, -maxdistance, maxdistance)
end

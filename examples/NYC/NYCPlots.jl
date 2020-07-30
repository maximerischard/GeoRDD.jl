using GeoInterface
using LibGEOS

using LinearAlgebra
using LaTeXStrings
using Formatting
import PyPlot; plt=PyPlot
cbbPalette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# using ..NYC: read_distr_reprpoints
using Distributions: MultivariateNormal
using Statistics: mean
using GaussianProcesses: predict_f
using PyCall
# mplot3d = pyimport("mpl_toolkits.mplot3d")

function plot_polygon(poly_coords, facecolor, alpha;
                      edgecolor="none", linestyle="None", linewidth=0.0, kwargs...)
    ax = plt.gca()
    poly_array = Array{Float64}(undef, length(poly_coords), 2)
    for i in 1:length(poly_coords)
        poly_array[i, :] = poly_coords[i]
    end
    polygon = plt.plt.Polygon(poly_array, true; facecolor=facecolor, kwargs...)
    p = ax.add_patch(polygon)
    p.set_edgecolor(edgecolor)
    p.set_linestyle(linestyle)
    p.set_linewidth(linewidth)
    p.set_alpha(alpha)
end

function plot_buffer(shape, border, color, dist, alpha; kwargs...)
    # a buffer zone around the border:
    border_buffer = LibGEOS.buffer(border, dist)
    # prevents a ring self-intersection bug (bad shapefile?):
    shape_rectified = LibGEOS.buffer(shape, 0.0)
    # the part of the buffer on the `shape` side of the border:
    shape_buffer = LibGEOS.intersection(border_buffer, shape_rectified)
    if typeof(shape_buffer) <: LibGEOS.Polygon
        poly_coords = GeoInterface.coordinates(shape_buffer)[1]
        plot_polygon(poly_coords, color, alpha; kwargs...)
    elseif typeof(shape_buffer) <: LibGEOS.MultiPolygon
        mpoly_coords = GeoInterface.coordinates(shape_buffer)
        for poly_coords in mpoly_coords
            plot_polygon(poly_coords[1], color, alpha; kwargs...)
        end
    else
        println(typeof(shape_buffer))
    end
end

function plot_schdistr_labels(;fontsize=12.0, labelcolor="black")
    plt.rc("text", usetex=false)
    # where to place the school district labels:
    schdistr_represent = Dict{SchDistr, Tuple{Float64, Float64}}(25 => (1035426.576356058, 215708.47750849032), 20 => (980957.6301213544, 166113.83169550286), 3 => (992826.0954164147, 226695.6766967253), 4 => (1000248.2836855762, 228298.3092956051), 16 => (1004214.0439845352, 188657.53201288398), 23 => (1008819.7413809987, 183460.7804870081), 18 => (1008787.3316119347, 173443.51080316756), 26 => (1054423.2425350049, 212717.9963988735), 32 => (1006895.1698167967, 192470.89239496225), 1 => (988952.4509786353, 202318.4309997029), 2 => (987095.1379395023, 210542.70410151145), 24 => (1013178.9222047399, 202219.43350214284), 28 => (1036851.1969587662, 195628.9613036562), 10 => (1008845.8110727321, 258295.0202941391), 9 => (1008189.8006258132, 245065.8830871096), 7 => (1006949.3221918349, 235790.44049067114), 30 => (1011393.8551870013, 216959.78259271727), 12 => (1017667.4294655474, 243924.23712153034), 8 => (1022714.7969213917, 237308.7890929639), 14 => (998121.9324939251, 199650.8938903253), 31 => (947923.9119948001, 147916.42111200799), 17 => (997069.2552085526, 180290.71820062993), 13 => (994552.5771594604, 190230.17111200746), 15 => (987418.7441204586, 181218.70071406086), 22 => (1001740.2660473757, 162368.14001459582), 21 => (991081.0983826279, 157637.68869012795), 11 => (1026344.6177574333, 256966.2814940873), 29 => (1055388.4397568335, 188398.91879267507), 19 => (1017942.166879338, 181833.5946349586), 27 => (1040711.4984081909, 178979.62564887386), 5 => (996548.0388532034, 237031.53149408224), 6 => (1001541.6608069859, 248002.3049010716))
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    ax = plt.gca()
    for schdistr in keys(schdistr_represent)
        center_x, center_y = schdistr_represent[schdistr]

        if !(xmin < center_x < xmax)
            continue
        elseif !(ymin < center_y < ymax)
            continue
        end

        ax.annotate(
            @sprintf("%d", schdistr),
            xy=(center_x, center_y),
            xycoords="data",
            xytext=(0,0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="black",
            color=labelcolor,
            alpha=1.0,
            fontsize = fontsize,
            zorder=3,
            annotation_clip=true,
        )
        ax.annotate(
            @sprintf("%d", schdistr),
            xy=(center_x, center_y),
            xycoords="data",
            xytext=(0,0),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="black",
            color=labelcolor,
            alpha=0.5,
            fontsize = fontsize,
            zorder=20,
            annotation_clip=true,
        )
    end
    plt.rc("text", usetex=true)
end

function plot_all_pairs(τpost_pairs::Dict{Tuple{SchDistr,SchDistr}}, regionData_dict::Dict{SchDistr,GeoRDD.RegionData}; scaleup=200.0, zorder="none")
    for distr_pair in keys(τpost_pairs)
        distrA, distrB = distr_pair

        τpost = τpost_pairs[distr_pair]
        # eff_size = abs(mean(τpost)) / std(τpost) * 2
        # if eff_size < 2.0
            # continue
        # end

        if distrA ∈ (1,2) || distrB ∈ (1,2)
            # don't do Manhattan, it's weird
            continue
        end

        # skip districts that aren't on the map
        if distrA ∉ (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32)
            continue
        end
        if distrB ∉ (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32)
            continue
        end

        if distrA >= distrB
            continue
        end

        shapeA = regionData_dict[distrA].shape
        shapeB = regionData_dict[distrB].shape
        border = GeoRDD.get_border(shapeA, shapeB, 10.0)

        border_coords = GeoInterface.coordinates(border)
        if typeof(border) <: LibGEOS.MultiLineString
        else
            border_coords = [border_coords]
        end
        ax = plt.gca()

        if mean(τpost) < 0
            cheaper_shape = shapeA
            pricier_shape = shapeB
        else
            cheaper_shape = shapeB
            pricier_shape = shapeA
        end
        plot_buffer(pricier_shape, border, cbbPalette[2], (abs(mean(τpost)) + std(τpost))*scaleup, 1.0, zorder=zorder-1)
        plot_buffer(pricier_shape, border, cbbPalette[1], abs(mean(τpost))*scaleup, 1.0, zorder=zorder)
    end
end

function plot_streets(;zoom=11, brighten=30, rgb=(200, 200, 200), zorder=-9)
    ax = plt.gca()
    ax.set_autoscale_on(false)
    ax.set_autoscalex_on(false)
    ax.set_autoscaley_on(false)
    EPSG=2263 # projection
    topleft =  (40.75, -74.01)
    botright = (40.58, -73.60)
    py"""
    import matplotlib.pyplot as plt

    import geopandas as gpd
    import tilemapbase
    import pyproj
    import PIL.Image as _Image

    def bounds(plotter, epsg):
        epsg_proj = pyproj.Proj('epsg:%d' % epsg, preserve_units=True)
        epsg_3857 = pyproj.Proj('epsg:%d' % tilemapbase.mapping._WEB_MERCATOR)
        self = plotter
        scale = 2 ** self.zoom
        x0, y0 = self.xtilemin / scale, self.ytilemin / scale # in tile space
        x1, y1 = (self.xtilemax + 1) / scale, (self.ytilemax + 1) / scale
        x0_ll, y0_ll = tilemapbase.to_lonlat(x0, y0)
        x1_ll, y1_ll = tilemapbase.to_lonlat(x1, y1)
        y0_ll, y1_ll = y1_ll, y0_ll # for some reason that I don't understand
        print(y0_ll)
        print(y1_ll)
        x0_epsg, y0_epsg = epsg_proj(x0_ll, y0_ll)
        x1_epsg, y1_epsg = epsg_proj(x1_ll, y1_ll)
        return x0_epsg, x1_epsg, y0_epsg, y1_epsg

    def brighten(img, factor):
        # split the image into individual bands
        source = img.split()
        R, G, B, A = 0, 1, 2, 3

        # process each band separately
        out_R = source[R].point(lambda x: min(x+factor, 256))
        out_G = source[G].point(lambda x: min(x+factor, 256))
        out_B = source[B].point(lambda x: min(x+factor, 256))

        # build a new multiband image
        brightened = _Image.merge(img.mode, (out_R, out_G, out_B, source[A]))
        return brightened

    def makecolor(img, rgb):
        # split the image into individual bands
        source = img.split()
        R, G, B, A = 0, 1, 2, 3

        # process each band separately
        out_R = source[R].point(lambda x: rgb[0])
        out_G = source[G].point(lambda x: rgb[1])
        out_B = source[B].point(lambda x: rgb[2])

        # build a new multiband image
        new_im = _Image.merge(img.mode, (out_R, out_G, out_B, source[A]))
        return new_im

    tilemapbase.start_logging()
    tilemapbase.init("/Users/imolk/tmp/tilemapbase.cache", create=True)

    extent = tilemapbase.Extent.from_lonlat($topleft[1], $botright[1],
                      $botright[0], $topleft[0])
    # plotter_ln = tilemapbase.Plotter(extent,tilemapbase.tiles.Carto_Dark_No_Labels, zoom=$zoom)
    plotter_ln = tilemapbase.Plotter(extent,tilemapbase.tiles.Stamen_Terrain_Lines, zoom=$zoom)
    plotter = plotter_ln
    # tile = brighten(plotter.as_one_image(), $brighten)
    tile = makecolor(plotter.as_one_image(), $rgb)
    $ax.imshow(tile, interpolation="lanczos", extent=bounds(plotter, $EPSG), origin="upper", zorder=$zorder)
    """
end

function ticks(xmin, xmax, tickdist)
    return tickdist * (ceil(xmin/tickdist):1.0:floor(xmax/tickdist))
end
function ticks_km(tickdist::Float64)
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    km_in_ft = 3280.84
    _xlim_km = _xlim ./ km_in_ft
    _ylim_km = _ylim ./ km_in_ft
    _xtick_km = ticks(_xlim_km[1], _xlim_km[2], tickdist)
    _ytick_km = ticks(_ylim_km[1], _ylim_km[2], tickdist)
    _xtick_ft = _xtick_km .* km_in_ft
    _ytick_ft = _ytick_km .* km_in_ft
    plt.xticks(_xtick_ft, [@sprintf("%.0f", x) for x in _xtick_km])
    plt.yticks(_ytick_ft, [@sprintf("%.0f", x) for x in _ytick_km])
    plt.xlabel("Eastings (km)")
    plt.ylabel("Northings (km)")
end

function plot_sales(filtered, missing_sqft; edgecolor="black", labelsize=12.0, labelcolor="black", colorbar=true, tickdist=5.0)
    fig=plt.figure()
    background_schdistrs(plt.gca(),
                color="#AAAAAA",
                edgecolor=edgecolor,
                alpha=1.0,
                linestyle="-",
                zorder=-10
                )
    background_schdistrs(plt.gca(),
                color="none",
                edgecolor=edgecolor,
                alpha=1.0,
                linestyle="-",
                linewidth=0.5,
                zorder=11)
    _xlim = (0.98e6, 1.06e6)
    _ylim = (1.5e5, 2.2e5)
    plt.plot(
        missing_sqft[!,:XCoord],
        missing_sqft[!,:YCoord],
        color="white",
        marker="x",
        markersize=1.0,
        linestyle="",
        zorder=8,
        alpha=0.7,
        linewidth=0.1,
        markeredgewidth=0.3,
    )
    plt.scatter(
        filtered[!,:XCoord], filtered[!,:YCoord],
        c=filtered[!,:logSalePricePerSQFT],
        cmap="jet",
        edgecolors="None",
        zorder=10,
        marker="o",
    #     vmin=4.0,
    #     vmax=8.0,
        alpha=1.0,
        s=1,
        )
    # plt.xlim(np.percentile(sales_geocoded_ggl["XCoord"].dropna(), [1,99]))
    # plt.ylim(np.percentile(sales_geocoded_ggl["YCoord"].dropna(), [1,99]))
    # plt.axis("off")
    # plt.title("log(Property Price per SQFT) in New York")
    ax = plt.gca()
    ax.set(aspect="equal", adjustable="box")
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    ticks_km(tickdist)

    ax.set_facecolor("#B0DAEE")
    if colorbar
        cbar = plt.colorbar(label="Square foot prices (\\\$ per sqft)")
        ax = cbar.ax
        tick_template = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        labeled = [10, 100, 1000, 10^4]
        ax.set_yticks(log.(labeled))
        ax.set_yticks(log.([tick_template; tick_template.*10; tick_template.*100; tick_template.*1000]),
                               minor=true)
        ax.set_yticklabels([format(x, commas=true) for x in labeled])
    end

    ax.set_autoscale_on(false)
    ax.set_autoscalex_on(false)
    ax.set_autoscaley_on(false)
    plot_schdistr_labels(fontsize=labelsize, labelcolor=labelcolor)
end
function plot_cliffface(μ, Σ, color; ndraws::Int=10, label=L"posterior of $\tau(x)$ on residuals")
    plt.plot(μ, color=color, ".", label=label)
    msize=10
    plt.plot(1,μ[1], color=cbbPalette[5],
        markersize=msize, marker="o")
    plt.plot(length(μ),μ[end], color=cbbPalette[4],
        markersize=msize, marker="o")
    posterior_sd = sqrt.(diag(Σ))
    plt.fill_between(1:length(μ), μ.-2*posterior_sd, μ.+2*posterior_sd,
        color=color,
        alpha=0.3,
        linewidth=0,
        zorder=-2,
        )
    posterior_distr = MultivariateNormal(μ, Σ)
    for _ in 1:ndraws
        plt.plot(1:length(μ), rand(posterior_distr), "-", color="white", alpha=0.4, linewidth=1.5, zorder=-1)
    end
    plt.xlim(1,length(μ))
    plt.ylim(minimum(μ.-2*posterior_sd), maximum(μ.+2*posterior_sd))

    # second y-axis for percentage price increase
    ax = plt.gca()
    y1, y2=ax.get_ylim()
    x1, x2=ax.get_xlim()
    ax2=ax.twinx()
    ax2.set_ylim(y1, y2)
    yticks_frac = collect(0.5: 0.1 : 2.0)
    yticks_log = log.(yticks_frac)
    in_window = y1 .< yticks_log .< y2
    ax2.set_yticks(yticks_log[in_window])
    ylabels = [@sprintf("%.0f\\%%", y*100) for y in yticks_frac[in_window]]
    ax2.set_yticklabels(ylabels)

    plt.sca(ax)
end

# plotting convenience functions
function yaxis_right()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
end
function title_in_axis(s, leftright)
    if leftright == :left
        plt.text(0.05, 0.95, s,
             horizontalalignment="left",
             verticalalignment="top",
             transform = plt.gca().transAxes)
    elseif leftright == :right
        plt.text(0.95, 0.95, s,
             horizontalalignment="right",
             verticalalignment="top",
             transform = plt.gca().transAxes)
    else
        throw("left or right?")
    end
end
function hide_xaxis()
    plt.gca().xaxis.set_ticklabels([])
    plt.xlabel("")
end
function pval_yticks()
    ax = plt.gca()
    plt.ylim(0,1)
    ax.set_yticks(0:0.1:1, minor=true)
    ax.set_yticks(0:0.2:1, minor=false)
end

# NYC_dir is GeoRDD/examples/NYC and is expected to be defined
# in the module that loads this file
NYSD_FILENAME = joinpath(NYC_dir, "NYC_data/nysd_16c/nysd.json")
py"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
plot_dataframe = gpd.plotting.plot_dataframe
import numpy as np
import itertools
plt.interactive(False)

EPSG=2263 # projection
# import os
# dirname = os.path.dirname(__file__)
# nysd_filename = os.path.join(dirname, "NYC_data/nysd_16c/nysd.json")
nycdistrs=gpd.read_file($(NYSD_FILENAME)).to_crs(epsg=EPSG)

def background_schdistrs(ax, **kwargs):
    plot_dataframe(nycdistrs, ax=ax, **kwargs)
    return None
"""
background_schdistrs = py"background_schdistrs"

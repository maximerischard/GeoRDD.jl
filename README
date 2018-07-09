# GeoRDD

[![Build Status](https://travis-ci.org/maximerischard/GeoRDD.jl.svg?branch=master)](https://travis-ci.org/maximerischard/GeoRDD.jl)

[![Coverage Status](https://coveralls.io/repos/maximerischard/GeoRDD.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/maximerischard/GeoRDD.jl?branch=master)

[![codecov.io](http://codecov.io/github/maximerischard/GeoRDD.jl/coverage.svg?branch=master)](http://codecov.io/github/maximerischard/GeoRDD.jl?branch=master)

This julia package implements the methods introduced in the paper **“A Bayesian Nonparametric Approach to Geographic Regression Discontinuity Designs: Do School Districts Affect NYC House Prices?”**

## What is a GeoRDD?

A GeoRDD is a natural experiment where units on one side of a geographical border are given a treatment while units on the other side are not. It's the spatial analog of univariate RDDs, which are an increasingly popular tool in econometrics and other social sciences.

## Quick Start

You first need to prepare a [dataframe](https://github.com/JuliaData/DataFrames.jl), for example with columns:
- `X1` and `X2` for the spatial covariates of the units,
- `outcome` for the outcome of interest
- `region` giving the treatment indicator (for example containing strings "treatment" and "control")
- `covarA` and `covarB` for two additional non-spatial real-valued covariates.
- `categ` for an additional categorical covariate

You also need the border coordinates as a
[LibGEOS](https://github.com/JuliaGeo/LibGEOS.jl) `LineString` or
`MultiLineString` object.
You can either create this yourself or read it from a shapefile or geojson file.
If what you have is polygons of the treatment and control regions, you can use
the provided `get_border` function to obtain the border between the two polygons.

With this in hand, you can use this package to obtain 
1. Estimates of the treatment effect everywhere along the border (the “cliff face” estimate),
2. Estimates of the local average treatment effect (LATE),
3. A \(p\)-value for the significance of the treatment effect.

See the
[`notebooks/SimulatedExample.jl`](notebooks/SimulatedExample.ipynb)
jupyter notebook for a step-by-step example of how to
perform the GeoRDD analysis.
If you run into any trouble, please open a GitHub issue.

For a more complete example using real data, see [`notebooks/SimulatedExample.jl`](notebooks/SimulatedExample.ipynb), which reproduces the example in the paper of estimating the difference in house prices between school districts in NYC.

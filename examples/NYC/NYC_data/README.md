# Sales data:

The Department of Finance’s Rolling Sales files lists properties that sold in the last twelve-month period in New York City for all tax classes. These files include:

* the neighborhood;
* building type;
* square footage;
* other data.

Download from (https://www1.nyc.gov/site/finance/taxes/property-annualized-sales-update.page)
These are provided in the `raw_data` directory. The data used in the manuscript is available in `raw_data/2015`.

## preprocessing the raw data

```bash
julia --project=Project.toml NYC_dataprep.jl --sales=NYC_data/raw_data/2015/2015_bronx.xls \
                      --sales=NYC_data/raw_data/2015/2015_brooklyn.xls \
                      --sales=NYC_data/raw_data/2015/2015_manhattan.xls \
                      --sales=NYC_data/raw_data/2015/2015_queens.xls \
                      --condos=NYC_data/DTM_Condo_Units.dbf \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Bronx_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Kings_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/NewYork_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Queens_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      > NYC_data/processed/NYC_sales_2015.csv
```

```bash
julia --project=Project.toml" NYC_dataprep.jl --sales=NYC_data/raw_data/2016/2016_bronx.xls \
                      --sales=NYC_data/raw_data/2016/2016_brooklyn.xls \
                      --sales=NYC_data/raw_data/2016/2016_manhattan.xls \
                      --sales=NYC_data/raw_data/2016/2016_queens.xls \
                      --condos=NYC_data/DTM_Condo_Units.dbf \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Bronx_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Kings_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/NewYork_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      --parcels=NYC_data/NYC-Tax-Parcels-Centroid-Points-SHP/Queens_2018_Tax_Parcel_Centroid_Points_SHP.shp \
                      > NYC_data/processed/NYC_sales_2016.csv
```

# School district boundaries

The shapes of the NYC school district boundaries for 2016 are in `nysd_16c/nysd.json` in
the geojson format.
Downloaded from https://www1.nyc.gov/site/planning/data-maps/open-data/districts-download-metadata.page

# NYC-Tax-Parcels-Centroid-Points-SHP

For every tax parcel, these ESRI shapefiles give the centroid of the lot as a tuple of projected X and Y coordinates. They can be downloaded from http://gis.ny.gov/gisdata/inventories/details.cfm?DSID=1300 . If needed, polygons of the tax lots can also be obtained from the digital tax map.

http://gis.ny.gov/gisdata/inventories/details.cfm?DSID=1300

# Digital Tax Map Condo units dbf file

Condominiums in NYC have a unique lot numbering system. To map them, we need the
digital tax map's condo units database, which can be downloaded (as a `.dbf`
file) as part of the Digital Tax Map from the Department of Finance at:
https://data.cityofnewyork.us/Housing-Development/Department-of-Finance-Digital-Tax-Map/smk3-tmxj/data

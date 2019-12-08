import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
plot_dataframe = gpd.plotting.plot_dataframe
import numpy as np
import itertools
plt.interactive(False)

EPSG=2263 # projection
import os
dirname = os.path.dirname(__file__)
nysd_filename = os.path.join(dirname, "NYC_data/nysd_16c/nysd.json")
nycdistrs=gpd.read_file(nysd_filename).to_crs(epsg=EPSG)

def background_schdistrs(ax, **kwargs):
    plot_dataframe(nycdistrs, ax=ax, **kwargs)
    return None

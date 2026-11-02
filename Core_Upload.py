# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 18:40:52 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj, Transformer
from scipy.interpolate import RegularGridInterpolator as RGI
import netCDF4 as nc
import xarray as xr 

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff

# Core Coordinates:
# Latitude: -79.480000* 
# Longitude: -112.110000

#%% Goff Bed Topo

Johns_bed_path = config.p(r"Modeling Data (Sticky Spot)/Beds/John Goff's Conditional Simulation/thwaitsbed_chan_consim4.grd")

# Load the .grd file
data = xr.open_dataset(Johns_bed_path)

# Access the variable
print(data)  # This will show the structure of the dataset
Z = data['z']  # Replace 'variable_name' with the actual name of the data variable
x_data = data['x']
y_data = data['y']

# Convert to numpy array if needed
Elev = Z.values
x = x_data.values
y = y_data.values

[X,Y] = np.meshgrid(x,y)

goff_bed_interp = RGI(
(x, y),              # Axis order is (Y, X)
Elev.T,              # Must match shape (len(Y), len(X))
method='linear',     # or 'cubic'
bounds_error=False,
fill_value=np.nan
)

# %% Bedmap Upload

bm3_surf_path = config.BM3_SURF

class bm3:
    
    surf = ff.new_tif_to_array(bm3_surf_path)

    surf_sub = surf[7200:8300,3200:4300].astype(float)
    surf_sub[surf_sub < -5000] = np.nan

    xc = np.arange(-3333500,3333500,500)
    yc = np.arange(3333500,-3333500,-500)

    [Xc, Yc] = np.meshgrid(xc,yc)
    
    y_sub = xc[3200:4300]
    x_sub = yc[7200:8300]
    
    X_sub, Y_sub = np.meshgrid(x_sub,y_sub)
    
    surf_interp = RGI(
    (y_sub, x_sub),          # Axis order is (Y, X)
    surf_sub.T,              # Must match shape (len(Y), len(X))
    method='linear',         # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )


# %%

crs_proj = "EPSG:3031"  # Antarctic Polar Stereographic, change if needed

# Transformer: WGS84 (lon/lat) -> projected (east/north)
transformer = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)

Core_x, Core_y = transformer.transform(-112.110000,-79.480000)

def read_core(file_path):
    
    import pandas as pd
    
    # Read into DataFrame
    df = pd.read_csv(file_path, sep="\t")
    
    # Preview the first few rows
    print(df.head())

    return df

df = read_core(config.WD2014)

print(list(df.columns))

depth = df['Depth ice/snow [m]']
calibrated_age = df['Cal age [ka BP] (ice age)']
# std_err = df['Cal age std e [±] (ice age uncertainty due to an...)']
std_err = df['Cal age std e [±] (ice age uncertainty due to CH...)']

plt.scatter(-calibrated_age,depth,c=-calibrated_age)
#plt.plot(-calibrated_age-std_err, depth)
#plt.plot(-calibrated_age+std_err, depth)
plt.xlabel('Calibrated Age [ka]')
plt.ylabel('Depth [m]')
ax = plt.gca()
ax.invert_xaxis()
ax.invert_yaxis()
# plt.xlim(-14,-16)
# plt.grid()


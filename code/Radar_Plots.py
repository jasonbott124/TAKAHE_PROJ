# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:07:04 2026

@author: Jason
"""

import config  # central data/output paths (config.py)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as RGI

df_names = [
    
    'Coherent Power',
    'Incoherent Power',
    'Power Ratio',
    'Specularity',
    'Specularity X',
    'Specularity Y'

    ]

df = { 
    'Coherent Power': [config.p(r"coherent_power.xyz")],
    'Incoherent Power': [config.p(r"incoherent_power.xyz")],
    'Power Ratio': [config.p(r"power_ratio.xyz")],
    'Specularity': [config.p(r"doi-10.18738-t8-c8ie7c/spec.xyz_val.grd")],
    'Specularity X': [config.p(r"specularityX.xyz_val.grd")],
    'Specularity Y': [config.p(r"specularityY.xyz_val.grd")]
    }


# Load raw X, Y, Z coordinate data

for i in df_names:
    file_path = df[i][0]

    try:
        # 1. Try reading it as a binary spatial grid (NetCDF/.grd) first
        # This handles the 0x89 binary files seamlessly
        grid_data = xr.open_dataarray(file_path)

        # Convert the spatial grid into a flat table of [X, Y, Z] rows
        df_flat = grid_data.to_dataframe().reset_index()
        df[i].append(df_flat.to_numpy())
        print(f"Successfully loaded binary grid file: {i}")

    except Exception:
        # 2. Fallback to reading it as a regular text file if xarray fails
        # Using encoding='latin-1' or errors='ignore' strips text corruption issues
        data = pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            comment=">",
            header=None,
            encoding="latin-1",
        )
        df[i].append(data.to_numpy())
        print(f"Successfully loaded text file: {i}")
        
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10), sharex=True, sharey=True)

axes = axes.flatten()

c = 0

ylims = [-6.2e5, -5.1e5]
xlims = [-1.45e6, -1.325e6]



for ax in axes:

    name = df_names[c]
    
    # Extract coordinates from the loaded data array at index [1]
    data_array = df[name][1]
    x = data_array[:, 0]
    y = data_array[:, 1]
    z = data_array[:, 2]
    
    # 2. Find the smallest and largest Z values
    z_min = np.nanmin(z)
    z_max = np.nanmax(z)
    
    n = 0.70
    cmin = z_min
    cmax = z_max
    
    scatter = ax.scatter(x, y, c=z, vmin=cmin, vmax=cmax, cmap='cividis', s=90, alpha=0.8)
    
    # 5. Add a colorbar specific to this subplot
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Power / Ratio Scale', fontsize=10)
    
    # 6. Customize subplot labeling
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    print(cmin, cmax)
    
    plt.tight_layout()

    c+=1
    
    if c > 2:
        break
    
    
def build_interpolator(x, y, z, decimals=6, method='linear'):
    """
    Turn flat (x, y, z) point-cloud columns from a regular grid file
    into a RegularGridInterpolator.
    """
    # Round to kill floating-point noise that would create bogus
    # "extra" unique coordinates (e.g. -1300000.0000001 vs -1300000.0)
    # xr_ = np.round(x, decimals)
    # yr_ = np.round(y, decimals)

    x_unique = np.unique(x)   # sorted, strictly ascending
    y_unique = np.unique(y)   # sorted, strictly ascending

    zgrid = np.full((len(y_unique), len(x_unique)), np.nan)

    xi = np.searchsorted(x_unique, x)
    yi = np.searchsorted(y_unique, y)
    zgrid[yi, xi] = z

    n_missing = np.isnan(zgrid).sum()
    if n_missing:
        print(f"  warning: {n_missing} / {zgrid.size} grid cells missing (holes in data)")

    interp = RGI(
        (y_unique, x_unique),   # axis order (Y, X)
        zgrid,                  # shape (len(Y), len(X)) — matches axis order, no .T needed
        method=method,
        bounds_error=False,
        fill_value= 0 
    )
    return interp


for i in df_names:
    data_array = df[i][1]
    x = data_array[:, 0]
    y = data_array[:, 1]
    z = data_array[:, 2]
    df[i].append(build_interpolator(x, y, z))
    

import pickle

interpolators = {name: df[name][2] for name in df_names}

with open("radar_return_interpolators.pkl", "wb") as f:
    pickle.dump(interpolators, f)
    

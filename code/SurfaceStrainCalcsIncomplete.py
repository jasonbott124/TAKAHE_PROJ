# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 19:38:38 2025

@author: Jason
"""
import config  # central data/output paths (config.py)
from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import netCDF4 as nc

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff

# %% Import Bed Topography Data

bm3_bed_path = config.BM3_BED
bm3_surf_path = config.BM3_SURF

class bm3:

    bed = ff.new_tif_to_array(bm3_bed_path)
    surf = ff.new_tif_to_array(bm3_surf_path)

    bed_sub = bed[7000:8600,3200:4500].astype(float)
    surf_sub = surf[7000:8600,3200:4500].astype(float)
    surf_sub[surf_sub < -5000] = np.nan

    xc = np.arange(-3333500,3333500,500)
    yc = np.arange(3333500,-3333500,-500)

    [Xc, Yc] = np.meshgrid(xc,yc)
    
    y_sub = xc[3200:4500]
    x_sub = yc[7000:8600]
    
    X_sub, Y_sub = np.meshgrid(x_sub,y_sub)
    
    surf_interp = RGI(
    (y_sub, x_sub),          # Axis order is (Y, X)
    surf_sub.T,              # Must match shape (len(Y), len(X))
    method='linear',         # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
    # bed_interp = RGI(
    # (x_sub, y_sub),  # Axis order is (Y, X)
    # bed_sub.T,            # Must match shape (len(Y), len(X))
    # method='linear',         # or 'cubic'
    # bounds_error=False,
    # fill_value=np.nan
    # )
    
    [X_sub,Y_sub] = np.meshgrid(x_sub,y_sub)

# %% Import Surface Velocity Data

Mv2 = nc.Dataset(config.MV2_VEL)

ylim = [np.min(bm3.X_sub), np.max(bm3.X_sub)]  # X-coordinate limits
xlim = [np.min(bm3.Y_sub), np.max(bm3.Y_sub)]   # Y-coordinate limits

x = Mv2['x'][:].data
y = Mv2['y'][:].data

xindex = np.where((x > xlim[0]) & (x < xlim[1]))
yindex = np.where((y > ylim[0]) & (y < ylim[1]))

x_filtered = x[xindex]
y_filtered = y[yindex]

X_filtered, Y_filtered = meshgrid(x_filtered,y_filtered)

vx_filtered = np.squeeze(Mv2['VX'][yindex][:, xindex].data)
vy_filtered = np.squeeze(Mv2['VY'][yindex][:, xindex].data)
errx_filtered = np.squeeze(Mv2['ERRX'][yindex][:, xindex].data)
erry_filtered = np.squeeze(Mv2['ERRY'][yindex][:, xindex].data)
stdx_filtered = np.squeeze(Mv2['STDX'][yindex][:, xindex].data)
stdy_filtered = np.squeeze(Mv2['STDY'][yindex][:, xindex].data)

erry_filtered[np.isnan(erry_filtered)] = 0
errx_filtered[np.isnan(errx_filtered)] = 0

fvx = RGI((x_filtered, y_filtered), vx_filtered.T, method='cubic', bounds_error=False)
fvy = RGI((x_filtered, y_filtered), vy_filtered.T, method='cubic', bounds_error=False)
ferrvx = RGI((x_filtered, y_filtered), errx_filtered.T, method='cubic', bounds_error=False)
ferrvy = RGI((x_filtered, y_filtered), erry_filtered.T, method='cubic', bounds_error=False)
fstdvx = RGI((x_filtered, y_filtered), stdx_filtered.T, method='cubic', bounds_error=False)
fstdvy = RGI((x_filtered, y_filtered), stdy_filtered.T, method='cubic', bounds_error=False)

# --- Flow magnitude and unit vectors ---
speed = np.sqrt(vx_filtered**2 + vy_filtered**2)
eps = 1e-12
mask = speed < eps

ux = np.full_like(vx_filtered, np.nan, dtype=float)
uy = np.full_like(vy_filtered, np.nan, dtype=float)
ux[~mask] = vx_filtered[~mask] / speed[~mask]
uy[~mask] = vy_filtered[~mask] / speed[~mask]

dy = 450
dx = 450

# %%

print('No. y = ', size(y_filtered))
print('No. x = ', size(x_filtered))

du,dv = gradient(speed, dy, dx)

DU = reshape(du, (len(x_filtered), len(y_filtered)))

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05    # e.g. 500 m/yr

plt.figure(figsize=(8,6))
plt.pcolormesh(X_filtered, Y_filtered, DU, shading="auto", cmap="viridis", 
               vmin=cbar_min, vmax=cbar_max)

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 15:15:00 2025

@author: Jason
"""

# Map Figure Maker

upload_decider = 0

# %% IMPORTS

import config  # central data/output paths (config.py)
import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)

import numpy as np
import uploader
import pandas as pd
import matplotlib.pyplot as plt
import Velocity_Normal_Vectors as vnv
import pyproj
import Flowband_Functions as ff
from  dataframe_merger_horizons import dataframe_combiner

# %% FILE PATHS
class Paths:
    
    bm3_bed_path = config.BM3_BED
    bm3_surf_path = config.BM3_SURF
    vel_path = config.MV2_VEL
    
    ffr_path_list = [
        
        config.p(r"hor_allpst/hor_allpst_FFR_surface_jbott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR1_JBott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR2_JBott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR3_JBott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR4_JBott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR5_JBott.csv"),
        config.p(r"hor_allpst/hor_allpst_FFR6_JBott.csv")
        
        ]
    
# %% DATA UPLOAD AND INITIALIZATION

ffr_dfs = {}

for i,path in enumerate(Paths.ffr_path_list):
    
    name = f'FFR{i}'
    df = pd.read_csv(path)
    ffr_dfs[name] = df

if upload_decider <= 0:
    # Bedmap v3
    xidx = [7000,8500]
    yidx = [3000,4500]
    bm3 = uploader.upload_bedmap3(Paths.bm3_bed_path, Paths.bm3_surf_path, xidx, yidx)
        
    # Mv2 Surface Velocity
    ylim = [np.min(bm3.X_sub), np.max(bm3.X_sub)]  # X-coordinate limits
    xlim = [np.min(bm3.Y_sub), np.max(bm3.Y_sub)]   # Y-coordinate limits
    Vel = uploader.upload_measures_sv(Paths.vel_path, ylim, xlim)
    
upload_decider += 1 # Uploading Bedmap takes a while, so this helps from accidentally uploading it twice

# %% SET DATA

xlims = [-1.55e6, -1.25e6]
ylims = [-6e5, -5e5]

Strain_list = vnv.Strain_Deliverer()

Strains = [
    
    Strain_list.lon_interp((Vel.Yc,Vel.Xc)),
    Strain_list.lat_interp((Vel.Yc,Vel.Xc)),
    Strain_list.shear_interp((Vel.Yc,Vel.Xc)),
    Strain_list.eff_interp((Vel.Yc,Vel.Xc))
    
    ]

strain_names = [
    
    'Longitudinal Strain Rate',
    'Lateral Strain Rate',
    'Shear Strain Rate',
    'Effective Strain Rate'
    
    ]

layer_names = [
    'FFR1',
    'FFR2',
    'FFR3',
    'FFR4',
    'FFR5',
    'FFR6'   
    ]

# %%

misc_pts_path = config.p(r"TAKAHE_data/Misc_Points/Interesting-Points-CSV.csv")

loc_df = pd.read_csv(misc_pts_path)

misc_pts_names = loc_df['Name']
misc_pts_X = loc_df['PS71_X']
misc_pts_Y = loc_df['PS71_Y']

from pyproj import Transformer

# 1. Choose your polar projection (e.g., EPSG:3413 for North Pole)
# Use "always_xy=True" to consistently accept (longitude, latitude) input order
transformer = Transformer.from_crs("epsg:3031", "epsg:4326", always_xy=True)

# 2. Convert a geographic point (Lon, Lat) to Polar Stereographic coordinates (X, Y in meters)
x, y = misc_pts_X, misc_pts_Y  # Directly at the North Pole
lon, lat = transformer.transform(x, y)

print(f"Projected X: {x}, Y: {y}")

# 3. Perform the inverse transformation (X, Y back to Lon, Lat)
inverse_transformer = Transformer.from_crs("epsg:3031", "epsg:4326", always_xy=True)
lon_out, lat_out = inverse_transformer.transform(x, y)

print(f"Geographic Lon: {lon_out}, Lat: {lat_out}")

# %% Strain Maps

# flights_path = 

# --- Plotting: 3 panels ---
fig, axs = plt.subplots(1, len(Strains), figsize=(18*5, 6*5), constrained_layout=True)

n_ticks = 3
i = 0

# ylims = [-7.0e5, -2.0e5]
# xlims = [-1.7e6, -1.2e6]

ylims = [-6.5e5, -4.7e5]
xlims = [-1.575e6, -1.25e6]

while i < len(Strains): 
    
    # Set your desired colorbar limits here:
    cbar_min = -0.08     # e.g. 0 m/yr
    cbar_max = 0.08  # e.g. 500 m/yr
    
    # Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
    ax = axs[i]
    
    if i == 0:
        ax.set_ylabel('Northings (EPS: 1301 Polar Stereographic)')
    
    if i !=3:
        pcm = ax.pcolormesh(Vel.Xc, Vel.Yc, Strains[i], shading='auto', cmap = 'seismic',
                             vmin=cbar_min, vmax=cbar_max)
        ax.grid(color = 'black', linestyle = '--', linewidth = 0.4)
        ax.scatter(misc_pts_X[10], misc_pts_Y[10], color = 'black')
    else:
        pcm = ax.pcolormesh(Vel.Xc, Vel.Yc, Strains[i], shading='auto', cmap = 'magma',
                             vmin=0.0, vmax=0.1)
        ax.grid(color = 'white', linestyle = '--', linewidth = 0.4)
        ax.scatter(misc_pts_X[10], misc_pts_Y[10], color = 'white')
        ax.scatter(misc_pts_X[14], misc_pts_Y[14], color = 'white')
    
    fig.colorbar(pcm, ax=ax, label='Strain Rate $s^{-1}$', orientation='horizontal', location='bottom', 
                 ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.03)
    
    # ax.plot(-1.387e6, -5.60e5, 'w^', markersize = 18)
    # ax.plot(-1.387e6, -5.602e5, 'r^', markersize = 12)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_xlabel('Eastings (EPS: 1301 Polar Stereographic)')
    
    # Transparent background
    # fig.patch.set_alpha(0)#0cd3f0
    # ax.set_facecolor('none')
    
    ax.set_title(strain_names[i], size = 30)
    ax.set_aspect('equal')
    
    i+=1
  
# %% Englacial Layer Map + VELOCITY

ylims = [-6.5e5, -4.7e5]
xlims = [-1.575e6, -1.25e6]

fig, axs = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)

i = 3

while i < len(ffr_dfs) - 1: 
    
    x = ffr_dfs[layer_names[i]]['x']
    y = ffr_dfs[layer_names[i]]['y']
    z = ffr_dfs[layer_names[i]]['z']
    
    # Set your desired colorbar limits here:
    cbar_min =  z.min()
    cbar_max = z.max()
    
    # Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
    ax = axs[i - 3]
    pcm = ax.scatter(x, y, c = z, cmap = 'viridis',
                         vmin=cbar_min, vmax=cbar_max)
    
    fig.colorbar(pcm, ax=ax, label='Elevation asl (m)', orientation='horizontal', location='bottom', 
                 ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.02)
    
    ax.scatter(-1388684, -560301, color = 'black')
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_title(layer_names[i])
    ax.set_aspect('equal')
    
    i+=1
    

# %% SURFACE TOPO, BED TOPO, & SURFACE VELOCITY

# bed_idx = ~np.isnan(bm3.bed_sub)
# surf_idx = ~np.isnan(bm3.surf_sub)

# bed = bm3.bed_sub[bed_idx]
# surf = bm3.surf_sub[surf_idx]

# %%

# 1. Choose your PS71 destination (e.g., EPSG:3995 for Arctic PS71)
# EPSG:3995 sets lat_ts=71 (latitude of standard parallel) and lon_0=0
# destination_crs = "epsg:3031" 

# # 2. Set up the transformer from WGS84 (lat/lon) to PS71
# transformer = Transformer.from_crs("epsg:4326", destination_crs, always_xy=True)

# # 3. Transform a point (Lon, Lat) 
# # Example: Longitude 15.0, Latitude 78.0 (Svalbard region)
# # lon_out, lat_out = 15.0, 78.0  
# misc_pts_X, misc_pts_Y = transformer.transform(lon_out, -lat_out)

# print(f"PS71 Projected X: {misc_pts_X:.2f} meters, Y: {misc_pts_Y:.2f} meters")

#%%
data_list = [
    
    bm3.bed_sub.T,
    Vel.speed,
    bm3.surf_sub.T
    
    ]

coord_list = [
    
    [bm3.Y_sub, bm3.X_sub],
    [Vel.Xc, Vel.Yc],
    [bm3.Y_sub, bm3.X_sub]
    
    ]

name_list = [
    
    'Bed Topography (Bedmap v3)',
    'Mv2 Surface Velocity (InSAR)',
    'Surface Topography (Bedmap v3)'
    
    ]

color_list = [
    
    'cmo.topo',
    'cmo.balance',
    'turbo'
    
    ]

lim_list = [
    
    [-2200, 2200],
    [0, 1200],
    [0, 2300]
    
    ]

size_magnf = 5

fig, axs = plt.subplots(1, 3, figsize=(18*size_magnf, 6*size_magnf), constrained_layout=False, sharey=False)

# Transparent background
fig.patch.set_alpha(0)#0cd3f0
ax.set_facecolor('none')

i = 0
n_ticks = 5

ylims = [-6.5e5, -4.7e5]
xlims = [-1.575e6, -1.25e6]

while i < 3: 
    
    # Set your desired colorbar limits here:
    # cbar_min = data_list[i][~np.isnan(data_list[i])].min() 
    # cbar_max = data_list[i][~np.isnan(data_list[i])].max() * 0.53
    
    cbar_min = lim_list[i][0]
    cbar_max = lim_list[i][1]
    
    Z = data_list[i]
    
    # Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
    ax = axs[i]
    pcm = ax.pcolormesh(coord_list[i][0], coord_list[i][1], data_list[i], cmap = color_list[i],
                         vmin=cbar_min, vmax=cbar_max)
    
    if name_list[i] == 'Mv2 Surface Velocity (InSAR)':
        cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                     ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
        contours = ax.contour(coord_list[i][0], coord_list[i][1], data_list[i], 
                              levels = np.linspace(cbar_min,350,n_ticks - 1), colors = 'white', alpha = 1)
        cbar.set_label('Surface Speed $ma^{-1}$', size = 11*size_magnf, **{'labelpad': 16})
        cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar
        
    elif name_list[i] == 'Surface Topography (Bedmap v3)':
        
        cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                     ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
        cbar.set_label('Elevation ASL (meters)', size = 11*size_magnf, **{'labelpad': 16})
        cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar
        
        contours = ax.contour(coord_list[i][0], coord_list[i][1], data_list[i], 
                              levels = np.linspace(cbar_min,cbar_max,n_ticks), colors = 'black', alpha = 1)
        
    else:
        cbar =  fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                     ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
        cbar.set_label('Elevation ASL (meters)', size = 11*size_magnf, **{'labelpad': 16})
        cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar
    
    if i == 0:
        ax.set_ylabel('Norhings (EPS:3031) $meters$', 
                      size = 10*size_magnf, labelpad = 20.0)
        ax.grid(color = 'black', linestyle = '--', linewidth = 0.3*size_magnf)
        ax.text(-1.362e6, -5.6e5, "Mount \n Takahe", color = 'black', size = 0.4*(size_magnf/4))
        ax.plot([-1.362e6, -1.387e6], [-5.6e5,-5.6012e5], color = 'black')
        
    elif i == 2:
        ax.grid(color = 'black', linestyle = '--', linewidth = 0.3*size_magnf)
        ax.text(-1.37e6, -5.7e5, "Mount \n Takahe", color = 'black', size = 0.4*size_magnf)
    
    else:
        ax.grid(color = 'white', linestyle = '--', linewidth = 0.3*size_magnf)
        ax.text(-1.378e6, -5.65e5, "Mount \n Takahe", color = 'white',size = 0.4*size_magnf)
        
    ax.set_xlabel('Eastings (EPS: 3031) $meters$', 
                  size = 10*size_magnf, labelpad = 20.0)
    
    # ax.scatter(misc_pts_X, misc_pts_Y)
    
    # ax.plot(-1.387e6, -5.60e5, 'w^', markersize = 25*size_magnf)
    # ax.plot(-1.387e6, -5.62e5, 'r^', markersize = 18*size_magnf)
    
    # White tick labels
    ax.tick_params(colors='black', size = 20, labelsize= 40)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_title(name_list[i], size = 10*size_magnf, pad = 30)
    ax.set_aspect('equal')
    
    # plt.tight_layout()
    
    ax.scatter(-1388684, -560301, color = 'white')
    ax.scatter(misc_pts_X, misc_pts_Y, color = 'white')
    
    i+=1
    
# %% JUST VELOCITY + RADAR LINES

size_magnf = 5

# ylims = [-6.5e5, -4.7e5]
# xlims = [-1.575e6, -1.25e6]

ylims = [-6.5e5, -2.25e5]
xlims = [-1.6e6, -1.2e6]

fig, axs = plt.subplots(1, 1, figsize=(10*size_magnf, 10*size_magnf), constrained_layout=False, sharey=False)

# Transparent background
fig.patch.set_alpha(0)#0cd3f0
ax.set_facecolor('none')

cbar_min = lim_list[1][0]
cbar_max = lim_list[1][1]

# Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
ax = axs
pcm = ax.pcolormesh(coord_list[1][0], coord_list[1][1], data_list[1], cmap = color_list[1],
                     vmin=cbar_min, vmax=cbar_max)


cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
             ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
contours = ax.contour(coord_list[1][0], coord_list[1][1], data_list[1], 
                      levels = np.linspace(cbar_min,350,n_ticks - 1), colors = 'white', alpha = 1)
cbar.set_label('Surface Speed $ma^{-1}$', size = 11*size_magnf, **{'labelpad': 16})
cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar

ax.set_xlabel('Eastings (EPS: 1301) $meters$', 
              size = 12*size_magnf, labelpad = 40.0)
ax.set_ylabel('Northings (EPS: 1301) $meters$', 
              size = 12*size_magnf, labelpad = 40.0)


ax.plot(-1.387e6, -5.59e5, 'w^', markersize = 25*size_magnf)
ax.plot(-1.387e6, -5.6e5, 'r^', markersize = 18*size_magnf)

# White tick labels
ax.tick_params(colors='white', size = 30, labelsize= 55)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.grid(color = 'white', linestyle = '--', linewidth = 0.2*size_magnf)
# ax.text(-1.378e6, -5.65e5, "Mount \n Takahe", color = 'white',size = 0.2*size_magnf)
        
# X47a
x47a_x = [-1388066, -1328065]
x47a_y = [-549916, -534291]

x43a_x = [-1369506, -1317260]
x43a_y = [-577184, -563614]

y15b_x = [-1397518, -1410386]
y15b_y = [-538077, -490743]

y31b_x = [-1386271, -1378669]
y31b_y = [-2.8e5, -3.1e5]

# y33a_x = 
# y33a_y = 

y17a_x = [-1426232, -1434803]
y17a_y = [-5.5e5, -5.2e5]

y19_x = [-1410494, -1424076]
y19_y = [-5.5e5, -5e5]

ax.plot(x47a_x, x47a_y, color = 'white', linewidth = 7, linestyle = '-', alpha = 0.6)
ax.plot(x43a_x, x43a_y, color = 'white', linewidth = 7, linestyle = '-', alpha = 0.6)
ax.plot(y15b_x, y15b_y, color = 'white', linewidth = 7, linestyle = '-', alpha = 0.6)
# ax.plot(y17a_x, y17a_y, color = 'white', linewidth = 7, linestyle = '-', alpha = 0.6)
ax.plot(y19_x, y19_y, color = 'white', linewidth = 7, linestyle = '-',  alpha = 0.6)
ax.plot(y31b_x, y31b_y, color = 'white', linewidth = 7, linestyle = '-')


ax.set_title(name_list[1], size = 10*size_magnf, pad = 30)
ax.set_aspect('equal')


# %% HORIZONS + TAKAHE

xlims = [-1.7e6,-0.5e6]
ylims = [-8e5, -1.5e5]

folder = config.p(r"THW_Muldoon_Leuro_Layers")

ml_horizons, name_list = ff.MuldoonLeuro_csv_reader(folder)

# layer_names = [
#     'FFR1',
#     'FFR2',
#     'FFR3',
#     'FFR4',
#     'FFR5',
#     'FFR6'   
#     ]

age_list = [
    
    'Isochron age: 0.548 +/- 0.2 ka',
    'Isochron age: 0.725 +/- 0.2 ka',
    'Isochron age: 4.711 +/- 0.246 ka',
    'Isochron age: 8.653 +/- 0.318 ka',
    'Isochron age: 17.177 +/- 0.413 ka',
    'Isochron age: 24.928 +/- 0.286 ka',
    
    ]

fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

i = 0
j = 0
k = 0

while i < 2: 
    
    j = 0
    
    while j < 3:
    
        x = ml_horizons[name_list[k]]['x_array']
        y = ml_horizons[name_list[k]]['y_array']
        z = ml_horizons[name_list[k]]['z_array']
        
        z[z < -10000] = np.nan # remove bad data values
        
        # Set your desired colorbar limits here:
        cbar_min =  z.min()
        cbar_max = z.max()
        
        # Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
        ax = axs[i,j]
        pcm = ax.scatter(x, y, c = z, cmap = 'viridis',
                              vmin=cbar_min, vmax=cbar_max)
        
        fig.colorbar(pcm, ax=ax, label='Elevation asl (m)', orientation='horizontal', location='bottom', 
                      ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.4, pad=0.02)
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        ax.set_title(age_list[k], pad = 10.0)
        ax.set_aspect('equal')
        
        ax.plot(-1.387e6, -5.60e5, 'w^', markersize = 18)
        ax.plot(-1.387e6, -5.602e5, 'r^', markersize = 12)
        
        ax.grid(color = 'white', linestyle = '--', linewidth = '0.4')
        
        j+=1
        k+=1
    
    i+=1
        

# %% HORIZONS + MAP + ONE COLOR SCALE

fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

i = 0
j = 0
k = 0

xlims = [-1.6e6, -0.8e6]
ylims = [-8e5, -1e5]

while i < 2: 
    
    j = 0
    
    while j < 3:
    
        x = ml_horizons[name_list[k]]['x_array']
        y = ml_horizons[name_list[k]]['y_array']
        z = ml_horizons[name_list[k]]['z_array']
        
        z[z < -10000] = np.nan # remove bad data values
        
        # Set your desired colorbar limits here:
        cbar_min =  z.min()
        cbar_max = z.max()
        
        # Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
        ax = axs[i,j]
        pcm = ax.scatter(x, y, c = z, cmap = 'viridis',
                              vmin=-1200, vmax=2400)
        
        fig.colorbar(pcm, ax=ax, label='Elevation asl (m)', orientation='horizontal', location='bottom', 
                      ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.4, pad=0.02)
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        ax.set_title(age_list[k], pad = 10.0)
        ax.set_aspect('equal')
        
        ax.plot(-1.387e6, -5.60e5, 'w^', markersize = 18)
        ax.plot(-1.387e6, -5.602e5, 'r^', markersize = 12)
        
        ax.tight_layout()
        
        ax.grid(color = 'white', linestyle = '--', linewidth = '0.4')
        
        j+=1
        k+=1
    
    i+=1
    
# %%

# FFR1 -> 2.24 ka
# FFR2 -> 0.64 ka



class Lists:
    
    list_1 = ml_horizons
    list_2 = ffr_dfs
    
    name_list = name_list
    age_list  = age_list 
    
    # Put ANY layer names here (ML layer keys + FFR keys), in the order you want:
    custom_order = [
        # Example: youngest -> oldest, mixing both sources
        "FFR1",
        name_list[0],
        "FFR2",
        name_list[1],
        "FFR3",
        name_list[2],
        "FFR4",
        name_list[3],
        "FFR5",
        name_list[4],
        "FFR6",
        name_list[5],
    ]
    
master_horizon_df = dataframe_combiner(Lists)

# %% HYDRO MAP

H = bm3.surf_sub - bm3.bed_sub   # Ice thickness
Zb = bm3.bed_sub                 # bed elevation

rho_ice = 917
rho_water = 1000

ylims = [-6.2e5, -5.1e5]
xlims = [-1.45e6, -1.325e6]
extents = [np.min(bm3.y_sub), np.max(bm3.y_sub), np.min(bm3.x_sub), np.max(bm3.x_sub)]

phi = rho_water*9.81*Zb + 9.81*H*rho_ice

d_phi = np.gradient(phi, 500, 500)

total_grad = np.sqrt(d_phi[0]**2 + d_phi[1]**2)

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(total_grad, vmin=0, vmax=np.nanpercentile(total_grad, 99), cmap='viridis', extent = extents
               , origin='upper')
plt.colorbar(im, ax=ax, label='|∇φ| (Pa/m)')

# --- highlight near-zero gradient (potential lake sites) ---
threshold = np.nanpercentile(total_grad, 2)  # tune this: "how flat counts as flat"

# contour now shares the same coordinate space as the imshow calls
ax.contour(total_grad, levels=[threshold], colors='red', linewidths=1.2, extent=extents, origin='upper')

flat_mask = np.ma.masked_where(total_grad > threshold, total_grad)
ax.imshow(flat_mask, cmap='autumn', vmin=0, vmax=threshold, extent=extents, origin='upper')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.scatter(misc_pts_X, misc_pts_Y, color = 'white')

plt.plot([-1337999, -1385831], [-567966, -581242])

plt.plot([-1381692, -1321352], [-596232, -579882])

plt.plot([-1347510,-1363878], [-603129, -542851])

plt.plot([-1362562, -1383039], [-607202, -531886])

ax.set_title('Hydraulic potential gradient — red = potential lake sites (flat φ)')
plt.show()

# %%

size_magnf = 5

fig, axs = plt.subplots(1, 3, figsize=(18*size_magnf, 6*size_magnf), constrained_layout=False, sharey=False)

# Transparent background
fig.patch.set_alpha(0)#0cd3f0
ax.set_facecolor('none')

i = 0
n_ticks = 5

ylims = [-6.5e5, -4.7e5]
xlims = [-1.575e6, -1.25e6]
    
# Set your desired colorbar limits here:
# cbar_min = data_list[i][~np.isnan(data_list[i])].min() 
# cbar_max = data_list[i][~np.isnan(data_list[i])].max() * 0.53

cbar_min = lim_list[i][0]
cbar_max = lim_list[i][1]

Z = data_list[i]

# Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
ax = axs[i]
pcm = ax.pcolormesh(coord_list[i][0], coord_list[i][1], data_list[i], cmap = color_list[i],
                     vmin=cbar_min, vmax=cbar_max)

if name_list[i] == 'Mv2 Surface Velocity (InSAR)':
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                 ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
    contours = ax.contour(coord_list[i][0], coord_list[i][1], data_list[i], 
                          levels = np.linspace(cbar_min,350,n_ticks - 1), colors = 'white', alpha = 1)
    cbar.set_label('Surface Speed $ma^{-1}$', size = 11*size_magnf, **{'labelpad': 16})
    cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar
    
elif name_list[i] == 'Surface Topography (Bedmap v3)':
    
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                 ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
    cbar.set_label('Elevation ASL (meters)', size = 11*size_magnf, **{'labelpad': 16})
    cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar
    
    contours = ax.contour(coord_list[i][0], coord_list[i][1], data_list[i], 
                          levels = np.linspace(cbar_min,cbar_max,n_ticks), colors = 'black', alpha = 1)
    ax.scatter(misc_pts_X, misc_pts_Y, color = 'white')

    ax.plot([-1337999, -1385831], [-567966, -581242])

    ax.plot([-1381692, -1321352], [-596232, -579882])

    ax.plot([-1347510,-1363878], [-603129, -542851])

    ax.plot([-1362562, -1383039], [-607202, -531886])
    
else:
    cbar =  fig.colorbar(pcm, ax=ax, orientation='horizontal', location='bottom', 
                 ticks = np.linspace(cbar_min,cbar_max,n_ticks), shrink=0.8, pad=0.07*(size_magnf/3))
    cbar.set_label('Elevation ASL (meters)', size = 11*size_magnf, **{'labelpad': 16})
    cbar.ax.tick_params(labelsize=40, size = 20) # For numbers/text on the colorbar

if i == 0:
    ax.set_ylabel('Norhings (EPS:1301) $meters$', 
                  size = 10*size_magnf, labelpad = 20.0)
    ax.grid(color = 'black', linestyle = '--', linewidth = 0.3*size_magnf)
    ax.text(-1.362e6, -5.6e5, "Mount \n Takahe", color = 'black', size = 0.4*(size_magnf/4))
    ax.plot([-1.362e6, -1.387e6], [-5.6e5,-5.6012e5], color = 'black')
    
elif i == 2:
    ax.grid(color = 'black', linestyle = '--', linewidth = 0.3*size_magnf)
    ax.text(-1.37e6, -5.7e5, "Mount \n Takahe", color = 'black', size = 0.4*size_magnf)

else:
    ax.grid(color = 'white', linestyle = '--', linewidth = 0.3*size_magnf)
    ax.text(-1.378e6, -5.65e5, "Mount \n Takahe", color = 'white',size = 0.4*size_magnf)
    
ax.set_xlabel('Eastings (EPS: 1301) $meters$', 
              size = 10*size_magnf, labelpad = 20.0)

# ax.plot(-1.387e6, -5.60e5, 'w^', markersize = 25*size_magnf)
# ax.plot(-1.387e6, -5.62e5, 'r^', markersize = 18*size_magnf)

# White tick labels
ax.tick_params(colors='white', size = 20, labelsize= 40)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_title(name_list[i], size = 10*size_magnf, pad = 30)
ax.set_aspect('equal')

# plt.tight_layout()

ax.scatter(-1388684, -560301, color = 'white')

i+=1




# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:33:14 2025

@author: Jason
"""

## Random Sample the Dataset to make it easier to plot in QGIS, Python, etc.

import config  # central data/output paths (config.py)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from scipy.interpolate import RegularGridInterpolator as RGI
import cmocean

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff
# import Horizon_CSV_Scraper as hs

from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
import uploader

# %%

bm3_bed_path = config.BM3_BED
bm3_surf_path = config.BM3_SURF
L1_path = config.p(r"Antarctica/layersforBodart/THW_MULDOON_DEEP_LAYER_1 - Copy.csv")
L3_path = config.p(r"THW_MULDOON_DEEP_LAYER_3 - Copy.csv")
L2_path = config.p(r"THW_MULDOON_DEEP_LAYER_2 - Copy.csv")
L4_path = config.p(r"THW_MULDOON_DEEP_LAYER_4 - Copy.csv")
S1_path = config.p(r"THW_LEURO_SHALLOW_1 - Copy.csv")
S2_path = config.p(r"THW_LEURO_SHALLOW_2 - Copy (2).csv")

# FFR_path = config.p("FFR_Horizons_WITH_SURFACE.csv")

ffr_path_list = [
    
    config.p(r"hor_allpst/hor_allpst_FFR_surface_jbott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR1_JBott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR2_JBott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR3_JBott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR4_JBott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR5_JBott.csv"),
    config.p(r"hor_allpst/hor_allpst_FFR6_JBott.csv")
    
    ]

ffr_dfs = {}

for i,path in enumerate(ffr_path_list):
    
    name = f'FFR{i}'
    df = pd.read_csv(path)
    ffr_dfs[name] = df
    
    
# %%

count = 20 # don't delete

# ffr_dfs = hs.horizon_scraper(FFR_path)

xidx = [7000,8500]
yidx = [3000,4500]

bm3 = uploader.upload_bedmap3(bm3_bed_path, bm3_surf_path, xidx, yidx)

# %%

vel_path = config.MV2_VEL
ylim = [np.min(bm3.X_sub), np.max(bm3.X_sub)]  # X-coordinate limits
xlim = [np.min(bm3.Y_sub), np.max(bm3.Y_sub)]   # Y-coordinate limits

Vel = uploader.upload_measures_sv(vel_path, ylim, xlim)

# %%

class L1:
    
    df = pd.read_csv(L1_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df.sample(n=int(np.size(df['x'])/count))
    
    # Get bounds from bm3 subregion
    y_min, y_max = bm3.x_sub.min(), bm3.x_sub.max()
    x_min, x_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    # # Generate regular grid
    # grid_x, grid_y = np.meshgrid(
    #     np.linspace(sample_df['x'].min(), sample_df['x'].max(), 200),
    #     np.linspace(sample_df['y'].min(), sample_df['y'].max(), 200)
    # )
    
    # # Interpolate using cubic spline
    # grid_z = griddata(
    #     points=(sample_df['x'], sample_df['y']),
    #     values=sample_df['Layer Depth'],
    #     xi=(grid_x, grid_y),
    #     method='linear')
    
# %%
class L3:
    
    df = pd.read_csv(L3_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df_new = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
class L2:
    
    df = pd.read_csv(L2_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df_new = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy() 
   
class L4: 
    
    df = pd.read_csv(L4_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df_new = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()


class S1:
    
    df = pd.read_csv(S1_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df_new = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    
class S2:
    
    df = pd.read_csv(S2_path)
    
    print(df.head())
    
    layer_depth = (df['thk_m'] + df['bedelv_mABSL']) - df['IRHdepth_m']
    x = df['x']
    y = df['y']
    
    df_new = pd.DataFrame({
        'Layer Depth': layer_depth,
        'x': x,
        'y': y
    })
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()

#%% 

class FFR_Surf:
    
    df_new = ffr_dfs['FFR0']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
       
    
class FFR1:
    
    df = ffr_dfs['FFR1']
    
    df['z'] = bm3.surf_interpolator((df['x'],df['y'])) - df['z']
    
    sample_df = df.sample(n=int(np.size(df['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= y_min) & (sample_df['x'] <= y_max) &
        (sample_df['y'] >= x_min) & (sample_df['y'] <= x_max)
    ].copy()
    
    # Check for NaNs
    # print("NaNs in y:", sample_df['y'].isna().sum())
    # print("NaNs in x:", sample_df['x'].isna().sum())
    # # Check for Infs
    # print("Infs in y:", np.isinf(sample_df['y']).sum())
    # print("Infs in x:", np.isinf(sample_df['x']).sum())
    
    # sample_df.to_csv("FFR1_sample_2.csv", index=False)
    
    # Generate regular grid
    # grid_x, grid_y = np.meshgrid(
    #     np.linspace(sample_df['x'].min(), sample_df['x'].max(), 200),
    #     np.linspace(sample_df['y'].min(), sample_df['y'].max(), 200)
    # )
    
    # # Interpolate using cubic spline
    # grid_z = griddata(
    #     points=(sample_df['x'], sample_df['y']),
    #     values=sample_df['z'],
    #     xi=(grid_x, grid_y),
    #     method='linear'
    # )
    
    
    
class FFR2:
    
    df_new = ffr_dfs['FFR2']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    # Interpolate the surface values
    surface_vals = bm3.surf_interp((sample_df['x'].values, sample_df['y'].values))
    
    sample_df = sample_df[sample_df['z'] > -200000]
    
class FFR3:
    
    df_new = ffr_dfs['FFR3']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    sample_df = sample_df[sample_df['z'] > -10000]
    
    
class FFR4:
    
    df_new = ffr_dfs['FFR4']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    sample_df = sample_df[sample_df['z'] > -10000]
    
class FFR5:
    
    
    df_new = ffr_dfs['FFR5']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
    # Interpolate the surface values
    surface_vals = bm3.surf_interp((sample_df['x'].values, sample_df['y'].values))
    
class FFR6:
    
    df_new = ffr_dfs['FFR6']
    
    sample_df = df_new.sample(n=int(np.size(df_new['x'])/count))
    
    # Get bounds from bm3 subregion
    x_min, x_max = bm3.x_sub.min(), bm3.x_sub.max()
    y_min, y_max = bm3.y_sub.min(), bm3.y_sub.max()
    
    # Filter sample_df to only include points within those bounds
    sample_df = sample_df[
        (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max) &
        (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    ].copy()
    
#%%

# # Create scatter plot
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(
#     sample_df['x'], 
#     sample_df['y'], 
#     c=sample_df['Layer Depth'], 
#     cmap='viridis',       # You can change to 'plasma', 'inferno', etc.
#     s=5,                   # size of each point
#     alpha=0.7              # transparency
# )

# # Add colorbar
# cbar = plt.colorbar(sc)
# cbar.set_label('Layer Depth (m)')

# # Axis labels
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Sampled Points Colored by Layer Depth')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))

# # Histogram of full data
# plt.hist(df_new['Layer Depth'], bins=100, alpha=0.5, label='Full Dataset', color='gray', density=True)

# # Histogram of sample data
# plt.hist(sample_df['Layer Depth'], bins=100, alpha=0.7, label='Sampled Dataset', color='blue', density=True)

# # Labels and legend
# plt.xlabel('Layer Depth (m)')
# plt.ylabel('Density')
# plt.title('Histogram of Layer Depth: Full Dataset vs Sample')
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%

# Example: your surface arrays
# x = bm3.x_sub
# y = bm3.y_sub
# X, Y = np.meshgrid(x,y)
# Z = bm3.surf_interp((X,Y))

# Vel.fvx_2D = RGI((Vel.Xc,Vel.Yc), Vel.x_filtered.T, method = 'linear')

# Vx = Vel.fvx_2D((Y,X))
# # vy = Vel.fvy((y,x))

# # speed = 0.5*np.sqrt(vx**2 + vy**2)

# plt.imshow(Vx)

# 2. Interpolate bed surface onto velocity grid
# coords = np.stack([Vel.Xc.ravel(), Vel.Yc.ravel()], axis=1)
# bm3_interpolated = bm3.surf_interp(Vel.x_filtered, Vel.y_filtered)

# 3. Plot using your custom Plotly function
# fig = ff.plot_Velocity(Z, speed, X=X, Y=Y, v_exag=0.2, cmocean_map = 'turbo', plot2D=False, contours=False)

# # fig = ff.plot_Surface_3D(bm3.interpolator((X,Y)),X,Y,x,y,DataType='Bed',cmapv = bm3.interpolator((X,Y)),plot2D=False, exag = 0.1)

# fig.show()

# 4. Static 2D plot for validation
# figure, ax = plt.subplots(1, figsize=(20, 12))
# im1 = ax.imshow(bm3_interpolated, cmap='viridis', vmin=-2000, vmax=4000, 
#                 extent=[bm3.x_sub.min(), bm3.x_sub.max(), 
#                         bm3.y_sub.min(), bm3.y_sub.max()])
# plt.colorbar(im1, ax=ax, label='Interpolated Bed Elevation (m)')
# ax.set_title('bm3_interpolated Surface')
# plt.show()

# %% Add the Layers

# Add sample points to the 3D plot
# fig.add_trace(go.Scatter3d(
#     x=FFR1.sample_df['x'],
#     y=FFR1.sample_df['y'],
#     z=FFR1.sample_df['z'],
#     mode='markers',
#     marker=dict(
#         size=3,
#         color= 'red',# S1.sample_df['Layer Depth'],    # Color by layer depth
#         colorscale='Viridis',
#         opacity=0.9,
#         colorbar=dict(title='Layer Depth')
#     ),
# ))

# fig.add_trace(go.Surface(
#     x=L1.grid_x,
#     y=L1.grid_y,
#     z=L1.grid_z + 500,
#     colorscale='Viridis',
#     opacity=0.85,
#     name='FFR1 Surface',
#     showscale=True,
#     colorbar=dict(title='Interpolated z')
# ))

# %%

# Add sample points to the 3D plot
fig.add_trace(go.Scatter3d(
    x=FFR1.sample_df['x'],
    y=FFR1.sample_df['y'],
    z=FFR1.sample_df['z'],
    mode='markers',
    marker=dict(
        size=3,
        color= 'green',#L2.sample_df['Layer Depth'],    # Color by layer depth
        # colorscale='Viridis',
        opacity=0.9,
        colorbar=dict(title='Layer Depth')
    ),
))

# Update layout for better visualization
# fig.update_layout(
#     title='test',
#     scene = dict(
#         xaxis_title='Eastings [m]',
#         yaxis_title='Northings [m]',
#         zaxis_title='test',
#         aspectratio=dict(x=1.3, y=1.0, z=0.03)
#     ),
#     autosize=True,
#        width=1920,   # or smaller/bigger depending on your screen size
#        height=1080,
#     margin=dict(l=45, r=50, b=65, t=100)
# )

# fig.show()

# # Add sample points to the 3D plot
# fig.add_trace(go.Scatter3d(
#     x=L3.sample_df['x'],
#     y=L3.sample_df['y'],
#     z=L3.sample_df['Layer Depth'],
#     mode='markers',
#     marker=dict(
#         size=3,
#         color= 'purple',#L3.sample_df['Layer Depth'],    # Color by layer depth
#         # colorscale='Viridis',
#         opacity=0.9,
#         colorbar=dict(title='Layer Depth')
#     ),
# ))

# # Add sample points to the 3D plot
# fig.add_trace(go.Scatter3d(
#     x=L4.sample_df['x'],
#     y=L4.sample_df['y'],
#     z=L4.sample_df['Layer Depth'],
#     mode='markers',
#     marker=dict(
#         size=3,
#         color= 'red',#L4.sample_df['Layer Depth'],    # Color by layer depth
#         # colorscale='Viridis',
#         opacity=0.9,
#         colorbar=dict(title='Layer Depth')
#     ),
# ))

#%%

# Example: your surface arrays
y = bm3.x_sub
x = bm3.y_sub
X, Y = np.meshgrid(x,y)
Z = bm3.surf_interp((X,Y))

# Compute gradients (slope)
dx, dy = np.gradient(Z)
slope_magnitude = np.sqrt(dx**2 + dy**2)

# Define light source direction (azimuth, elevation)
azimuth = 315  # NW
altitude = 45  # mid-sun
az = np.deg2rad(azimuth)
alt = np.deg2rad(altitude)

# Surface normals (approx)
nx = -dx
ny = -dy
nz = np.ones_like(Z)

# Normalize vectors
norm = np.sqrt(nx**2 + ny**2 + nz**2)
nx, ny, nz = nx/norm, ny/norm, nz/norm

# Light vector
lx = np.cos(alt) * np.cos(az)
ly = np.cos(alt) * np.sin(az)
lz = np.sin(alt)

# Illumination (dot product)
illum = nx*lx + ny*ly + nz*lz
illum = np.clip(illum, 0, 1)  # keep between 0 and 1

# Plot surface with hillshade
# fig = go.Figure()

fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z,
    surfacecolor=illum,                   # use illumination for shading
    colorscale=[[0, 'white'], [1, 'white']],  # grayscale hillshade
    showscale=False,
    opacity = 1.0,
    name = "Hillshaded Surface"
))

# # Example: also add bed fully colored
# fig.add_trace(go.Surface(
#     x=bm3.X_sub,
#     y=bm3.Y_sub,
#     z=bm3.bed_sub,
#     colorscale='Viridis',
#     showscale=True,
#     opacity=1.0,
#     name="Bed"
# ))

fig.update_layout(
    scene=dict(
        xaxis_title="Eastings [m]",
        yaxis_title="Northings [m]",
        zaxis_title="Elevation [m]"
    )
)


# fig.add_trace(go.Surface(
#     x=bm3.X_sub[0:605000],
#     y=bm3.Y_sub[0:605000],
#     z=bm3.surf_sub[0:605000], 
#     surfacecolor=np.ones_like(bm3.surf_sub),  # force uniform color
#     colorscale=[[0, 'white'], [1, 'white']],  # all white
#     showscale=False,  # hide colorbar
#     opacity=1.0,
#     name='Surface'
# ))

# # Update layout for better visualization
# fig.update_layout(
#     title='test',
#     scene = dict(
#         xaxis_title='Eastings [m]',
#         yaxis_title='Northings [m]',
#         zaxis_title='test',
#         aspectratio=dict(x=1.3, y=1.0, z=0.1)
#     ),
#     autosize=True,
#        width=1920,   # or smaller/bigger depending on your screen size
#        height=1080,
#     margin=dict(l=45, r=50, b=65, t=100)
# )

# Update layout with axis limits
fig.update_layout(
    title='test',
    scene=dict(
        # xaxis=dict(title='Eastings [m]', range=[-5.5e5, -6.5e5]),
        # yaxis=dict(title='Northings [m]', range=[-1.52e6, -1.37e6]),
        # zaxis=dict(title='test', range=[-4000, 4000]),  # adjust as needed
        aspectratio=dict(x=1.0, y=1.0, z=0.1)
    ),
    autosize=True,
    width=1920,
    height=1080,
    margin=dict(l=45, r=50, b=65, t=100)
)

fig.show()

# scene=dict(
#     xaxis=dict(range=[-5e5, -4e5]),
#     yaxis=dict(range=[-1.55e6, -1.45e-6])
# )

# fig.write_html("bm3_surf_with_muldoon_deep.html", full_html=True, include_plotlyjs='cdn')

# fig.show()

# fig = go.Figure(data=[go.Scatter3d(x=sample_df['x'], y=sample_df['y'], z=sample_df['Layer Depth'], mode='markers',
#                                         marker=dict(
#                                             size=5,
#                                             color=sample_df['Layer Depth'],    # Color by layer depth
#                                             colorscale='Viridis',
#                                             opacity=1.0,
#                                             colorbar=dict(title='Layer Depth')
#                                         ),)])

# import webbrowser
# import os

# file_path = os.path.abspath("bm3_surf_with_muldoon_deep.html")
# webbrowser.open('file://' + file_path)

# print("Bed min/max:", np.nanmin(bm3.bed_sub), np.nanmax(bm3.bed_sub))
# print("Layer Depth min/max:", sample_df['Layer Depth'].min(), sample_df['Layer Depth'].max())

# print("x range (sample):", sample_df['x'].min(), sample_df['x'].max())
# print("x range (bm3):", bm3.x_sub.min(), bm3.x_sub.max())
# print("y range (sample):", sample_df['y'].min(), sample_df['y'].max())
# print("y range (bm3):", bm3.y_sub.min(), bm3.y_sub.max())


# plt.show()




'''
Header Rows:

0 - Mystery Horizon
217433 - FFR1
486251 - FFR2
2209 - FFR3
15940 - FFR4
23030 - FFR5
115959 - FFR6
1004696 - FFR_surface

'''

#%%

import matplotlib.pyplot as plt

# Extract the DataFrame for FFR1
df = ffr_dfs['FFR1']
surf_df = ffr_dfs['FFR0']

# Basic 2D scatter plot: x vs y, color by z
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['x'], df['y'], c = bm3.surf_interp((df['x'],df['y'])) - df['z'], cmap='viridis', s=10)
plt.colorbar(sc, label='z value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('FFR1: 2D Scatter Colored by z')
plt.grid(True)
plt.tight_layout()
plt.ylim(-3e5,-7e5)
plt.xlim(-1.1e6, -1.5e6)
plt.show()
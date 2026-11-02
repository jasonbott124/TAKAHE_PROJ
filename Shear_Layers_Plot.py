# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 19:07:51 2025

@author: Jason
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:01:27 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
from scipy.interpolate import RegularGridInterpolator as RGI
import netCDF4 as nc
import xarray as xr

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
from stat_melt import statistical_melt 
import layer_shape_functions as lsf

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Velocity_Normal_Vectors as vnv

# %%

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Core_Upload as cu

bm3_surf_path = config.BM3_SURF
bm3_bed_path = config.BM3_BED

Johns_bed_path = config.p(r"Modeling Data (Sticky Spot)/Beds/John Goff's Conditional Simulation/thwaitsbed_chan_consim4.grd")

#%% Goff Bed Topo

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
# class bm3:
    
#     sub_idxs = [7000,8800,2800,5000]
    
#     surf = ff.new_tif_to_array(bm3_surf_path)
#     bed = ff.new_tif_to_array(bm3_bed_path)

#     surf_sub = surf[sub_idxs[0]:sub_idxs[1],sub_idxs[2]:sub_idxs[3]].astype(float)
#     surf_sub[surf_sub < -5000] = np.nan
    
#     bed_sub = bed[sub_idxs[0]:sub_idxs[1],sub_idxs[2]:sub_idxs[3]].astype(float)
#     bed_sub[bed_sub < -5000] = np.nan

#     xc = np.arange(-3333500,3333500,500)
#     yc = np.arange(3333500,-3333500,-500)

#     [Xc, Yc] = np.meshgrid(xc,yc)
    
#     y_sub = xc[sub_idxs[2]:sub_idxs[3]]
#     x_sub = yc[sub_idxs[0]:sub_idxs[1]]
    
#     X_sub, Y_sub = np.meshgrid(x_sub,y_sub)
    
#     surf_interp = RGI(
#     (y_sub, x_sub),          # Axis order is (Y, X)
#     surf_sub.T,              # Must match shape (len(Y), len(X))
#     method='linear',         # or 'cubic'
#     bounds_error=False,
#     fill_value=np.nan
#     )
    
#     bed_interp = RGI(
#     (y_sub, x_sub),          # Axis order is (Y, X)
#     bed_sub.T,              # Must match shape (len(Y), len(X))
#     method='linear',         # or 'cubic'
#     bounds_error=False,
#     fill_value=np.nan
#     )
    
# plt.imshow(bm3.bed_sub)

if "bm3" in globals():
    del bm3
    import gc
    gc.collect()

def load_bedmap3(bm3_bed_path, bm3_surf_path, sub_idxs):
    
    surf = ff.new_tif_to_array(bm3_surf_path)
    bed  = ff.new_tif_to_array(bm3_bed_path)

    surf_sub = surf[sub_idxs[0]:sub_idxs[1], sub_idxs[2]:sub_idxs[3]].astype(float)
    surf_sub[surf_sub < -5000] = np.nan

    bed_sub = bed[sub_idxs[0]:sub_idxs[1], sub_idxs[2]:sub_idxs[3]].astype(float)
    bed_sub[bed_sub < -5000] = np.nan

    xc = np.arange(-3333500, 3333500, 500)
    yc = np.arange( 3333500,-3333500,-500)

    y_sub = xc[sub_idxs[2]:sub_idxs[3]]
    x_sub = yc[sub_idxs[0]:sub_idxs[1]]
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

    surf_interp = RGI(
        (y_sub, x_sub), surf_sub.T,
        bounds_error=False, fill_value=np.nan
    )

    bed_interp = RGI(
        (y_sub, x_sub), bed_sub.T,
        bounds_error=False, fill_value=np.nan
    )

    return {
        "surf_sub": surf_sub,
        "bed_sub": bed_sub,
        "X_sub": X_sub,
        "Y_sub": Y_sub,
        "surf_interp": surf_interp,
        "bed_interp": bed_interp
    }

bm3 = load_bedmap3(bm3_bed_path, bm3_surf_path, [6700,8400,3100,5200])
plt.imshow(bm3["bed_sub"])
    
#%% Measures Velocity Upload

Mv2 = nc.Dataset(config.MV2_VEL)

ylim = [np.min(bm3["X_sub"]), np.max(bm3["X_sub"])]  # X-coordinate limits
xlim = [np.min(bm3["Y_sub"]), np.max(bm3["Y_sub"])]   # Y-coordinate limits

x = Mv2['x'][:].data
y = Mv2['y'][:].data

xindex = np.where((x > xlim[0]) & (x < xlim[1]))
yindex = np.where((y > ylim[0]) & (y < ylim[1]))

class Vel:
    
    x_filtered = x[xindex]
    y_filtered = y[yindex]
    
    Xc,Yc = np.meshgrid(x_filtered,y_filtered)

    vx_filtered = np.squeeze(Mv2['VX'][yindex][:,xindex].data)
    vy_filtered = np.squeeze(Mv2['VY'][yindex][:,xindex].data)
    errx_filtered = np.squeeze(Mv2['ERRX'][yindex][:,xindex].data)
    erry_filtered = np.squeeze(Mv2['ERRY'][yindex][:,xindex].data)
    stdx_filtered = np.squeeze(Mv2['STDX'][yindex][:,xindex].data)
    stdy_filtered = np.squeeze(Mv2['STDY'][yindex][:,xindex].data)
    
    erry_filtered[np.isnan(erry_filtered)] = 0 # Sets all NaN values to zero, we'll see if this fixes things
    errx_filtered[np.isnan(errx_filtered)] = 0
    
    fvx = RGI((x_filtered, y_filtered), vx_filtered.T, method='cubic', bounds_error=False)
    fvy = RGI((x_filtered, y_filtered), vy_filtered.T, method='cubic', bounds_error=False)
    
    ferrvx = RGI((x_filtered, y_filtered), errx_filtered.T, method='cubic', bounds_error=False)
    ferrvy = RGI((x_filtered, y_filtered), erry_filtered.T, method='cubic', bounds_error=False)
    
    fstdvx = RGI((x_filtered, y_filtered), stdx_filtered.T, method='cubic', bounds_error=False)
    fstdvy = RGI((x_filtered, y_filtered), stdy_filtered.T, method='cubic', bounds_error=False)
    
    v_sub = np.sqrt(vx_filtered**2 + vy_filtered**2)

# %% Find Point of maximum negative slope at each layer??

# %%

# --- SETTINGS ---
folder = config.p(r"Margin Layers/X47a_Fixed_Maybe") # folder containing horizon txt files
crs_proj = "EPSG:3031"  # Antarctic Polar Stereographic, change if needed

# Transformer: WGS84 (lon/lat) -> projected (east/north)
transformer = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)

# --- STORAGE ---
all_layers = {}  # dictionary of arrays keyed by filename

for file in os.listdir(folder):
    
    if not file.endswith(".txt"):
        continue

    filepath = os.path.join(folder, file)

    # Read horizon file
    df = pd.read_csv(filepath, header=None)
    # Columns: 0=label, 1,2=index(?), 3=lon, 4=lat, 5=elevation
    df.columns = ["label", "col1", "col2", "lon", "lat", "elev"]

    # Convert lat/lon -> projected coords
    east, north = transformer.transform(df["lon"].values, df["lat"].values)
    
    # df_xyz[f"rel_{colname[2:]}"] = (df_xyz[colname] - df_xyz['bed']) / (df_xyz['srf'] - df_xyz['bed'])

    # Store arrays
    x_array = np.array(east)
    y_array = np.array(north)
    z_array = df["elev"].values

    # Compute cumulative distance along track
    dx = np.diff(x_array)
    dy = np.diff(y_array)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    x_position = np.concatenate([[0], np.cumsum(segment_lengths)])

    all_layers[file] = {
        "x_array": x_array,
        "y_array": y_array,
        "z_array": z_array,
        "x_position": x_position,
    }
    
    plt.scatter(y_array,-z_array)
    # plt.xlim(-1.05e6,-1.09e6)


# %% Pick bed_file as the layer with the most data points
base_file, bed_layer = max(
    all_layers.items(),
    key=lambda item: len(item[1]["z_array"])
)

# base_file = 'Y35a_EM9_jbott_THW_SJB2_Y35a.txt'

surface_file = r"Surface_THW_SJB2_X47a.txt"
bed_file =  r"Bed_THW_SJB2_X47a.txt"

# str.ls('*surface*')


print("Chosen bed_file (most points):", base_file)

surf_file = list(all_layers.keys())[1]
print("Example layer:", base_file)
print("x_position:", all_layers[base_file]["x_position"])
print("z_array:", all_layers[base_file]["z_array"])

base_x = all_layers[base_file]["x_array"]
base_y = all_layers[base_file]["y_array"]

surface = bm3['surf_interp']((base_x,base_y))
bed = goff_bed_interp((base_x,base_y))
surf_vel = np.sqrt(Vel.fvx((base_x,base_y))**2 + Vel.fvy((base_x,base_y))**2)

plt.style.use('dark_background') 
fig, ax = plt.subplots(figsize=(8,16))
# ax.plot(all_layers[base_file]["y_array"],bed,'w')
# ax.plot(all_layers[base_file]["y_array"], surface, 'w')

surface = all_layers[surface_file]['z_array']
surface_x = all_layers[surface_file]['x_array']
surface_y = all_layers[surface_file]['y_array']
surface_pos = all_layers[surface_file]['x_position']

bed = all_layers[bed_file]['z_array']
bed_x = all_layers[bed_file]['x_array']
bed_y = all_layers[bed_file]['y_array']
bed_pos = all_layers[bed_file]['x_position']

surf_interp = RGI(
(surface_y,),              # Axis order is (Y, X)
surface,             # Must match shape (len(Y), len(X))
method='linear',     # or 'cubic'
bounds_error=False,
fill_value=np.nan
)

bed_interp = RGI(
(bed_y,),              # Axis order is (Y, X)
bed,             # Must match shape (len(Y), len(X))
method='linear',     # or 'cubic'
bounds_error=False,
fill_value=np.nan
)


ax = plt.gca()
for fname, layer in list(all_layers.items()):
    
    x_pos = layer["y_array"]
    rel_surf = surf_interp((x_pos))
    rel_bed = bed_interp((x_pos))
    # x_pos = layer["x_array"]
    z_arr = (layer["z_array"] - rel_bed) / (rel_surf - rel_bed)
    ax.scatter(x_pos, z_arr, label=fname)
    
    
#x_limits = (-1.05e6,-1.09e6) 
#plt.xlim(x_limits)
# plt.ylim(1, 0)
# plt.gca().invert_yaxis() # Invert the y-axis
#plt.ylim(-2500, 1500)
    
# ax.fill_between(all_layers[base_file]["y_array"],bed, y2=ax.get_ylim()[0], color='gray', alpha=0.4)
# ax.fill_between(all_layers[base_file]["y_array"],bed, y2=surface, color='blue', alpha=0.2)

#%%

Strains = vnv.Strain_Deliverer()

# %%

# plt.plot(all_layers[base_file]["y_array"],surf_vel)
# plt.xlim(-5.55e5, -5.2e5)

# Need to add variable basal B value

#figure;
# % plot(Grid.p.xc + Grid.p.dx,tau_bx); hold on
# % plot(Grid.p.xc,Rxy_h)
# % plot(Grid.p.xc,Re)
# % plot(Grid.p.xc, Data.p.Srf_V)
# plot(Grid.p.xc, melt_rate*1000*s2yr); hold on
#title('Analytical Melt Rate at Shear Margin')
#xlabel('Distance Along Profile'); ylabel('Melt Rate (mm/yr)')  
# contourf(Grid.p.Xc,Grid.p.Yc,dtdy);

# %%

folder = config.p(r"THW_Muldoon_Leuro_Layers")

ml_horizons, name_list = ff.MuldoonLeuro_csv_reader(folder)

age_list = [
    
    'Isochron age: 0.548 +/- 0.2 ka',
    'Isochron age: 0.725 +/- 0.2 ka',
    'Isochron age: 4.711 +/- 0.246 ka',
    'Isochron age: 8.653 +/- 0.318 ka',
    'Isochron age: 17.177 +/- 0.413 ka',
    'Isochron age: 24.928 +/- 0.286 ka'
    
    ]

# %%

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

x_548 = ml_horizons['THW_LEURO_SHALLOW_1 - Copy.csv']['x_array']
y_548 = ml_horizons['THW_LEURO_SHALLOW_1 - Copy.csv']['y_array']
z_548 = ml_horizons['THW_LEURO_SHALLOW_1 - Copy.csv']['z_array']

# Unstructured data points: (x, y, value)
# points = np.array(x_548,y_548)
values = np.array(z_548)

# Create the interpolator
interp_548 = LinearNDInterpolator(((x_548,y_548)), values)

x_725 = ml_horizons['THW_LEURO_SHALLOW_2 - Copy (2).csv']['x_array']
y_725 = ml_horizons['THW_LEURO_SHALLOW_2 - Copy (2).csv']['y_array']
z_725 = ml_horizons['THW_LEURO_SHALLOW_2 - Copy (2).csv']['z_array']

# Unstructured data points: (x, y, value)
# points = np.array(x_725,y_725)
values = np.array(z_725)

interp_725 = LinearNDInterpolator(((x_725,y_725)), values)


# %% Pick bed_file as the layer with the most data points

# base_file = 'Y31b_EM9_jbott_THW_SJB2_Y27a.txt'

# print("Chosen bed_file (most points):", base_file)

# surf_file = list(all_layers.keys())[1]
# print("Example layer:", base_file)
# print("x_position:", all_layers[base_file]["x_position"])
# print("z_array:", all_layers[base_file]["z_array"])
# print('\n')

# print("Chosen bed_file (most points):", base_file)

# surf_file = list(all_layers.keys())[1]
# print("Example layer:", base_file)
# print("x_position:", all_layers[base_file]["x_position"])
# print("z_array:", all_layers[base_file]["z_array"])

# surface = bm3.surf_interp((base_x,base_y))
# bed = goff_bed_interp((base_x,base_y))
# Thickness= surface - bed

# X47a_labels = [
    
#     'Bedrock',
#     '2.240845 ka', # ka
#     '0.635773 ka',  # ka
#     'Ice Surface'
          
#     ]

# def frac_depth_plot(Files, Interpolators, Arguments, Data, Strains):
    
#     import Core_Upload as cu
    
#     all_layers = Data.all_layers
#     Vel = Data.Vel
    
#     base_x = all_layers[Files.base_file]["x_array"]
#     base_y = all_layers[Files.base_file]["y_array"]
    
#     bed_file = Files.bed_file
#     surf_file = Files.surf_file
    
#     B = Interpolators.real_bed_interp
#     S = Interpolators.real_surf_interp
    
#     if Arguments.direction == 'x':
        
#         a = 'x'
#         b = 'y'
#         vel_pos = base_x
        
#         # print(vel_pos)
        
#         surface_x = all_layers[surf_file]['x_array']
#         surface_y = all_layers[surf_file]['y_array']
        
#         bed_x = all_layers[bed_file]['x_array']
#         bed_y = all_layers[bed_file]['y_array']
    
#     else:
        
#         a = 'y'
#         b = 'x'
#         vel_pos = base_y
        
#         # print(vel_pos)
        
#         surface_x = all_layers[surf_file]['y_array']
#         surface_y = all_layers[surf_file]['x_array']
        
#         bed_x = all_layers[bed_file]['y_array']
#         bed_y = all_layers[bed_file]['x_array']
        
    
#     surface = all_layers[surf_file]['z_array']
#     # surface_pos = all_layers[surf_file]['x_position']

#     bed = all_layers[bed_file]['z_array']
#     # bed_pos = all_layers[bed_file]['x_position']

#     surf_interp = RGI(
#     (surface_x,),              # Axis order is (Y, X)
#     surface,             # Must match shape (len(Y), len(X))
#     method='linear',     # or 'cubic'
#     bounds_error=False,
#     fill_value=np.nan
#     )

#     bed_interp = RGI(
#     (bed_x,),              # Axis order is (Y, X)
#     bed,             # Must match shape (len(Y), len(X))
#     method='linear',     # or 'cubic'
#     bounds_error=False,
#     fill_value=np.nan
#     )
    
#     surf_vel = np.sqrt(Vel.fvx((base_x,base_y))**2 + Vel.fvy((base_x,base_y))**2)
#     lat_strain = Strains.lat_interp((base_y,base_x))
#     eff_strain = Strains.eff_interp((base_y,base_x))
    
#     # x_sorted_indices = np.argsort(x_unsorted)
#     # y_sorted_indices = np.argsort(y_unsorted)

#     # x_sorted = x_unsorted[x_sorted_indices]
#     # y_sorted = y_unsorted[y_sorted_indices]

#     plt.style.use('dark_background') 

#     # --- Create two stacked plots with shared x-axis ---
#     fig, (ax_vel, ax_strain, ax_tau, ax_layers) = plt.subplots(
#     4, 1, figsize=(20, 10),
#     gridspec_kw={'height_ratios': [1, 1, 1, 2]}  # must be length 3
#     )
    
#     # --- Velocity plot (top) ---
#     ax_vel.scatter(vel_pos, surf_vel, c = surf_vel, cmap = 'cmo.balance'
#                    , vmin = 0, vmax = 150)
#     ax_vel.set_ylabel("Velocity (m/yr)", color='w')
#     ax_vel.tick_params(axis='y', labelcolor='w')

#     if Arguments.depth == 'Fractional':
        
#         for fname, layer in list(all_layers.items()):
            
#             x_pos = layer[f"{a}_array"]
#             rel_surf = surf_interp((x_pos))
#             rel_bed = bed_interp((x_pos))
#             # x_pos = layer["x_array"]
#             z_arr = (layer["z_array"] - rel_bed) / (rel_surf - rel_bed)
            
#             if Arguments.plot_style == 'scatter':
                
#                 ax_layers.scatter(x_pos, z_arr, label=fname, s = Arguments.point_size)
#                 # ax_layers.scatter(base_x, interp_548((base_x,base_y)),label= '0.548 ka +/- 0.02 ka', s = Arguments.point_size)
                
#             elif Arguments.plot_style == 'line':
                
#                 ax_layers.plot(x_pos, z_arr, label=fname)
                
#             else:
#                 ax_layers.scatter(x_pos, z_arr, label=fname, s = Arguments.point_size)
                
#                 # ax_layers.scatter(cu.Core_y*np.ones(np.size(cu.depth)), S((cu.Core_x,cu.Core_y)) - cu.depth, c = cu.calibrated_age, 
#                                   # vmin = cu.calibrated_age.min(), vmax = cu.calibrated_age.max(), cmap = 'viridis')
                
#     elif Arguments.depth == 'Elevation':
        
#         i = 0
        
#         for fname, layer in list(all_layers.items()):
            
#             x_pos = layer[f"{a}_array"]
#             y_pos = layer[f"{b}_array"]
#             rel_surf = surf_interp((x_pos))
#             rel_bed = bed_interp((x_pos))
            
#             x = layer['x_array']
#             y = layer['y_array']
            
#             real_bed = B((x,y))
#             real_surface = S((x,y))
#             real_thk = real_surface - real_bed
            
#             # x_pos = layer["x_array"]
#             z_arr = (rel_surf - layer["z_array"]) / (rel_surf - rel_bed)
#             z_arr = real_surface - (z_arr * real_thk)
            
#             if Arguments.plot_style == 'scatter':
                
#                 # ax_layers.scatter(x_pos, z_arr, label=X47a_labels[i], s = Arguments.point_size)
                
#                 i+=1
                
#             elif Arguments.plot_style == 'line':
                
#                 ax_layers.plot(x_pos, z_arr, label=fname)
                
#             else:
#                 ax_layers.scatter(x_pos, z_arr, label=fname, s = Arguments.point_size)
                
#         # FFR2_x = all_layers['FFR1_JBott_THW_SJB2_X47a.txt']['x_array']
#         # FFR2_y = all_layers['FFR1_JBott_THW_SJB2_X47a.txt']['y_array']
        
#         # idx = np.where(FFR2_x > -1.378e6)
        
#         # usable_x = FFR2_x[idx]
#         # usable_y = FFR2_y[idx]
            
#         ax_layers.plot(surface_x, S((surface_x,surface_y)), label = 'Ice Surface')
#         # ax_layers.plot(usable_x, interp_548((usable_x,usable_y)),label= '0.548 ka +/- 0.02 ka')
#         # ax_layers.plot(usable_x, interp_725((usable_x,usable_y)),label= '0.725 ka +/- 0.02 ka')
        
#         # ax_layers.set_ylim(-1700, 1600)
        
#         sc = ax_layers.scatter(cu.Core_y*np.ones(np.size(cu.depth)), S((cu.Core_x,cu.Core_y)) - cu.depth, c = cu.calibrated_age, 
#                           vmin = cu.calibrated_age.min(), vmax = cu.calibrated_age.max(), cmap = 'viridis')
        
#         cbar = fig.colorbar(sc, ax=ax_layers)
#         cbar.set_label("Calibrated Age")
        
#     else:
#          return "ERROR: Please specify type of z-data to plot: 'Fractional' or 'Elevation' "
     
     
#     print("layers plotted")
     
#     theta = np.zeros(np.size(surf_vel))
     
#     # ramp section: between 30 and 85
#     mask = (surf_vel > Arguments.low_thresh) & (surf_vel < Arguments.upper_thresh)
#     theta[mask] = (surf_vel[mask] - Arguments.low_thresh) / (Arguments.upper_thresh - Arguments.low_thresh)
    
#     # print(theta.max(), theta.min())
    
#     Thickness = S((base_x,base_y)) - B((base_x,base_y))
    
#     s2yr = (60**2)*24*365.25;
    
#     # print(Thickness, S((base_x,base_y)), B((base_x,base_y)))
    
#     tau_b = Arguments.Bo *(((Arguments.m*Arguments.n + 1)/(2*Thickness))*(surf_vel)*(1 - theta))**(1/Arguments.n); # basal shear stress
    
#     G = 0.06; dTdz = 0.02; k = 2.2; rho = 917;L = 3.34e5
    
#     u_b = surf_vel * theta
    
#     m = (1/(rho*L)) * (G - k*(dTdz) + (1/s2yr)*tau_b*u_b)
    
#     melt_in_mm = m*s2yr*1000
    
#     print('Melt Calculated')

#     ax_tau.scatter(vel_pos, melt_in_mm, c = melt_in_mm, cmap = 'cmo.thermal',
#                    vmin = 0, vmax = 7)
#     ax_tau.set_ylabel("Melt Rate", color='w')
#     ax_tau.tick_params(axis='y', labelcolor='w')
    
#     ax_vel.set_ylim(0,50)
    
#     surf_vel = np.sqrt(Vel.fvx((base_x,base_y))**2 + Vel.fvy((base_x,base_y))**2)
#     lat_strain = Strains.lat_interp((surface_y,surface_x))
#     eff_strain = Strains.eff_interp((base_y,base_x))
    
#     ax_strain.scatter(vel_pos, lat_strain, c = lat_strain, cmap = 'seismic', vmin = -0.04, vmax = 0.04)
#     ax_strain.set_ylabel("Lateral Strain $s^{-1}$", color='w')
#     ax_strain.tick_params(axis='y', labelcolor='w')
#     ax_strain.set_xlim(Arguments.limits)
#     ax_strain.set_ylim(Arguments.strain_lims)
    
#     # Configure labels and x limits for layer plot
#     ax_layers.set_xlim(Arguments.limits)
#     ax_layers.set_xlabel("EPS 1301 Eastings (meters)")
#     ax_layers.set_ylabel("Elevation ASL (meters)")
    
#     for ax in (ax_vel, ax_tau, ax_layers):
#         ax.set_xlim(Arguments.limits)
    
#     # Transparent background
#     fig.patch.set_alpha(0)#0cd3f0
#     ax_layers.set_facecolor('none')
#     ax_strain.set_facecolor('none')
#     ax_tau.set_facecolor('none')
#     ax_vel.set_facecolor('none')
    
#     plt.tight_layout()
    
#     # Grid on the layer plot
#     ax_layers.grid(color = 'white', linestyle = '--', linewidth = 0.4)
#     plt.legend(loc = 'lower right')
    
#     # fill between bed and surface, as well as bed and ymin. Helps give the layer plot some life.
#     ax = plt.gca()
#     ax_layers.fill_between(surface_x, B((surface_x,surface_y)), 
#                            S((surface_x,surface_y)), color='blue', alpha = 0.2)
#     ax_layers.fill_between(surface_x, -1700, B((surface_x,surface_y)), color='gray', alpha = 0.4)
    
#     ax_layers.set_ylim(-3000, 4000)
    
#     # ax_layers.set_aspect('equal')
    
#     plt.show()
    
#     return bed_interp, surf_interp
    

'''
Variables needed to be specified:
    base_file for velocity (probably surface works best)
    surface file and bed file
    overall direction (x or y)
    
Things to add:
    Bedmap (INSAR) surface elevation
    Lateral Stress
'''

# %%

# Cleaned + fixed version of frac_depth_plot
#
# Key fixes / improvements:
# - Removes unused vars + duplicate computations
# - Consistent axis handling (direction 'x' or 'y')
# - Uses 1D interpolation safely (np.interp) instead of RGI for 1D profiles
# - Robust sorting for interpolation axes
# - Avoids reusing "ax" incorrectly and avoids plt.gca() surprises
# - Consolidates styling, limits, transparency, legend, grids
# - Clear helper functions and early validation
#
# Assumptions:
# - Data.all_layers[fname] has keys: 'x_array','y_array','z_array'
# - Vel has callables: Vel.fvx((x,y)), Vel.fvy((x,y))
# - Interpolators has callables: real_bed_interp((x,y)), real_surf_interp((x,y))
# - Strains has: lat_interp((y,x)), eff_interp((y,x))  (note order)
# - Arguments has: direction, depth, plot_style, point_size,
#                  low_thresh, upper_thresh, Bo, m, n,
#                  limits, strain_lims
#
# Optional: if you rely on cmocean colormaps, import cmocean before calling.

import numpy as np
import matplotlib.pyplot as plt

def frac_depth_plot(Files, Interpolators, Arguments, Data, Strains):
    """
    Plots:
      1) surface velocity along a chosen profile
      2) lateral strain along the same profile
      3) modeled melt rate proxy (melt_in_mm)
      4) radar layers in either fractional depth or elevation space

    Returns
    -------
    bed_profile_interp : callable
        1D interpolator for bed elevation vs along-track coordinate
    surf_profile_interp : callable
        1D interpolator for surface elevation vs along-track coordinate
    """
    
    palette = ["#D8DBE2", "#A9BCD0", "#58A4B0", "#373F51", "#1B1B1E"]
    TEXT_COLOR  = palette[0]
    LIGHT_FILL  = palette[1]
    LIGHT_EDGE  = palette[2]
    DARK_LINE   = palette[3]


    # -------------------------
    # Helpers
    # -------------------------
    def _require(cond, msg):
        if not cond:
            raise ValueError(msg)

    def _sorted_interp1d(x, z, fill=np.nan):
        """
        Returns a function f(xq) that linearly interpolates z(x).
        Handles unsorted x and repeated points (keeps first occurrence).
        """
        x = np.asarray(x).astype(float)
        z = np.asarray(z).astype(float)

        # sort
        idx = np.argsort(x)
        x = x[idx]
        z = z[idx]

        # drop NaNs
        good = np.isfinite(x) & np.isfinite(z)
        x = x[good]
        z = z[good]

        # drop duplicate x (np.interp requires increasing x)
        if x.size == 0:
            return lambda xq: np.full_like(np.asarray(xq, dtype=float), fill, dtype=float)

        uniq_x, uniq_idx = np.unique(x, return_index=True)
        uniq_z = z[uniq_idx]

        def f(xq):
            xq = np.asarray(xq, dtype=float)
            out = np.interp(xq, uniq_x, uniq_z, left=np.nan, right=np.nan)
            out[~np.isfinite(out)] = fill
            return out

        return f

    def _compute_speed(Vel, x, y):
        vx = Vel.fvx((x, y))
        vy = Vel.fvy((x, y))
        return np.sqrt(vx * vx + vy * vy)

    def _set_transparent(fig, axes):
        fig.patch.set_alpha(0)
        for a in axes:
            a.set_facecolor("none")

    # -------------------------
    # Validate inputs
    # -------------------------
    all_layers = Data.all_layers
    Vel = Data.Vel

    _require(hasattr(Files, "base_file"), "Files.base_file missing")
    _require(hasattr(Files, "bed_file"), "Files.bed_file missing")
    _require(hasattr(Files, "surf_file"), "Files.surf_file missing")
    _require(Files.base_file in all_layers, f"base_file not in all_layers: {Files.base_file}")
    _require(Files.bed_file in all_layers, f"bed_file not in all_layers: {Files.bed_file}")
    _require(Files.surf_file in all_layers, f"surf_file not in all_layers: {Files.surf_file}")
    _require(Arguments.direction in ("x", "y"), "Arguments.direction must be 'x' or 'y'")
    _require(Arguments.depth in ("Fractional", "Elevation"),
             "Arguments.depth must be 'Fractional' or 'Elevation'")

    # Base track arrays (full 2D path coordinates)
    base_x = np.asarray(all_layers[Files.base_file]["x_array"])
    base_y = np.asarray(all_layers[Files.base_file]["y_array"])

    # 1D coordinate used for plotting along-track
    # If direction == 'x' => horizontal axis is x, otherwise y
    along_key = "x" if Arguments.direction == "x" else "y"
    across_key = "y" if along_key == "x" else "x"
    vel_pos = base_x if along_key == "x" else base_y

    # Profile layers for bed/surface used to build *1D* bed/surface along-track interpolators
    bed_layer = all_layers[Files.bed_file]
    surf_layer = all_layers[Files.surf_file]

    bed_along = np.asarray(bed_layer[f"{along_key}_array"])
    bed_z = np.asarray(bed_layer["z_array"])

    surf_along = np.asarray(surf_layer[f"{along_key}_array"])
    surf_z = np.asarray(surf_layer["z_array"])

    # Real interpolators (2D) for absolute elevation conversion
    B2D = Interpolators.real_bed_interp
    S2D = Interpolators.real_surf_interp

    # Build robust 1D interpolators along-track
    bed_profile_interp = _sorted_interp1d(bed_along, bed_z, fill=np.nan)
    surf_profile_interp = _sorted_interp1d(surf_along, surf_z, fill=np.nan)

    # -------------------------
    # Compute along-track fields
    # -------------------------
    surf_speed = _compute_speed(Vel, base_x, base_y)  # m/yr

    # strain interpolators appear to want (y, x)
    lat_strain_along = Strains.lat_interp((base_y, base_x))
    eff_strain_along = Strains.eff_interp((base_y, base_x))

    # -------------------------
    # Figure layout
    # -------------------------
    plt.style.use("dark_background")

    fig, (ax_vel, ax_strain, ax_tau, ax_layers) = plt.subplots(
        4, 1, figsize=(60, 30),
        gridspec_kw={"height_ratios": [1, 1, 1, 2]},
        sharex=True
    )
    
    fig.set_dpi(150)

    # Transparent background for powerpoint overlay
    _set_transparent(fig, [ax_vel, ax_strain, ax_tau, ax_layers])

    # -------------------------
    # 1) Velocity
    # -------------------------
    
    vel_lims = getattr(Arguments, "vel_lims", [np.min(surf_speed)*1.05,0.9*np.max(surf_speed)])
    
    ax_vel.scatter(
        vel_pos, surf_speed,
        c=surf_speed, cmap="viridis",
        vmin=vel_lims[0], vmax=vel_lims[1],
        s=max(5, getattr(Arguments, "point_size", 10))
    )
    # ax_vel.set_ylabel("Velocity (m/yr)", size = 40, labelpad = 35)
    ax_vel.set_ylim(vel_lims[0]*0.8, vel_lims[1]*1.2)

    # -------------------------
    # 2) Strain (lateral)
    # -------------------------
    
    strain_lims = getattr(Arguments, "strain_lims", [-0.05,0.05])
    
    ax_strain.scatter(
        vel_pos, lat_strain_along,
        c=lat_strain_along, cmap="seismic",
        vmin=strain_lims[0], vmax=strain_lims[1],
        s=max(5, getattr(Arguments, "point_size", 10))
    )
    # ax_strain.set_ylabel(r"Lateral Strain $s^{-1}$", size = 40, labelpad = 35)
    ax_strain.set_ylim(Arguments.strain_lims)

    # -------------------------
    # 3) Melt proxy (tau_b, melt_in_mm)
    # -------------------------
    
    melt_mode = getattr(Arguments, "melt_mode", "deterministic")  # "deterministic" or "statistical"

    thickness = S2D((base_x, base_y)) - B2D((base_x, base_y))
    # theta ramp from speed thresholds
    theta = np.zeros_like(surf_speed, dtype=float)
    mask = (surf_speed > Arguments.low_thresh) & (surf_speed < Arguments.upper_thresh)
    denom = (Arguments.upper_thresh - Arguments.low_thresh)
    if denom <= 0:
        raise ValueError("Arguments.upper_thresh must be > Arguments.low_thresh")
    theta[mask] = (surf_speed[mask] - Arguments.low_thresh) / denom
    theta[surf_speed >= Arguments.upper_thresh] = 1.0

    if melt_mode == "statistical":
        # Feed per-point arrays into the Monte Carlo
        Arguments.Thickness = thickness
        Arguments.surf_vel  = surf_speed
        Arguments.theta     = theta
    
        melt_mc_mm, _ = statistical_melt(Arguments)     # melt_mc_mm shape: (N, M)

        i = 0
        while i < int(Arguments.N):
            ax_tau.plot(vel_pos, melt_mc_mm[i], alpha = 0.2, color = palette[2])
            i += 10
            
        mean = np.mean(melt_mc_mm, axis = 0)
        std = np.std(melt_mc_mm, axis=0)
        
        ax_tau.fill_between(
            vel_pos, mean + 2*std, mean - 2*std,
            color=palette[1],
            alpha=0.35,
            label=r"2$\sigma$"
        )

        ax_tau.plot(
            vel_pos, mean,
            color= palette[3],
            linewidth=3.5,
            label="Mean"
        )
        
        print(mean)

        # plt.grid(linestyle = '--', linewidth = 0.4, color = 'white')

        # # Adjust font, markers, and spacing
        ax_tau.legend(loc='upper right', fontsize=30, markerscale=3.0,
                  labelspacing=1.8, handlelength=3.5, borderpad=1.0, facecolor=palette[3])
        
        melt_lims = getattr(Arguments, "melt_lims", [-15,15])
        
        ax_tau.set_ylim(melt_lims[0],melt_lims[1])
    
    else:
    
        # basal shear stress (your equation retained)
        tau_b = Arguments.Bo * (
            ((Arguments.m * Arguments.n + 1) / (2 * thickness)) * (surf_speed) * (1 - theta)
        ) ** (1 / Arguments.n)
    
        # melt model terms (kept as-is, but cleaned)
        s2yr = (60 ** 2) * 24 * 365.25
        G, dTdz, k, rho, L = 0.06, 0.02, 2.2, 917.0, 3.34e5
    
        u_b = surf_speed * theta
        m = (1 / (rho * L)) * (G - k * dTdz + (1 / s2yr) * tau_b * u_b)  # m/s
        melt_in_mm = m * s2yr * 1000  # mm/yr
    
        ax_tau.scatter(
            vel_pos, melt_in_mm,
            c=melt_in_mm, cmap="cmo.thermal",
            vmin=0, vmax=7,
            s=max(5, getattr(Arguments, "point_size", 10))
        )
    
        # ax_tau.set_ylabel("Melt rate (mm/yr)", size = 40, labelpad = 35)

    # -------------------------
    # 4) Layer plot
    # -------------------------
    def _plot_layer(along_vals, z_vals, label):
        if Arguments.plot_style == "line":
            ax_layers.plot(along_vals, z_vals, label=label)
        else:
            ax_layers.scatter(along_vals, z_vals, s=Arguments.point_size, label=label)

    if Arguments.depth == "Fractional":
        # fractional depth = (z - bed) / (surf - bed)
        for fname, layer in all_layers.items():
            along_vals = np.asarray(layer[f"{along_key}_array"])
            z = np.asarray(layer["z_array"])

            rel_surf = surf_profile_interp(along_vals)
            rel_bed = bed_profile_interp(along_vals)

            denom = (rel_surf - rel_bed)
            frac = (z - rel_bed) / denom
            frac[~np.isfinite(frac)] = np.nan

            _plot_layer(along_vals, frac, fname)

        # ax_layers.set_ylabel("Fractional depth (0=bed, 1=surface)")

    elif Arguments.depth == "Elevation":
        # Convert each layer to real elevation using 2D bed/surf thickness
        for fname, layer in all_layers.items():
            x = np.asarray(layer["x_array"])
            y = np.asarray(layer["y_array"])
            along_vals = np.asarray(layer[f"{along_key}_array"])

            # local 1D surface/bed for fractional position (from profile)
            rel_surf = surf_profile_interp(along_vals)
            rel_bed = bed_profile_interp(along_vals)

            # fraction measured downward from surface
            frac_down = (rel_surf - np.asarray(layer["z_array"])) / (rel_surf - rel_bed)
            frac_down[~np.isfinite(frac_down)] = np.nan

            real_bed = B2D((x, y))
            real_surf = S2D((x, y))
            real_thk = real_surf - real_bed

            z_real = real_surf - frac_down * real_thk
            _plot_layer(along_vals, z_real, fname)

        # optional: plot actual surface along the same along-track coordinate
        surf_across = np.asarray(surf_layer[f"{across_key}_array"])
        # reconstruct (x,y) along the surface layer for plotting S2D
        if along_key == "x":
            surf_x, surf_y = surf_along, surf_across
        else:
            surf_x, surf_y = surf_across, surf_along

        ax_layers.plot(surf_along, S2D((surf_x, surf_y)), label="Ice Surface", linewidth = 5)

        # ax_layers.set_ylabel("Elevation ASL (m)", size = 40, labelpad = 35.0)

    # -------------------------
    # Shared formatting
    # -------------------------
    for ax in (ax_vel, ax_strain, ax_tau, ax_layers):
        ax.set_xlim(Arguments.limits)
        ax.grid(color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.tick_params(labelsize=40, size = 20, pad = 20)

    ax_layers.set_xlabel("EPSG:3031 along-track coordinate (m)", size = 60, labelpad = 40.0)

    # Fill between bed and surface for context (use the surface-layer coordinates)
    if along_key == "x":
        fill_x = surf_along
        fill_y = np.asarray(surf_layer["y_array"])
        bed_fill = B2D((fill_x, fill_y))
        surf_fill = S2D((fill_x, fill_y))
    else:
        fill_x = surf_along
        fill_y = np.asarray(surf_layer["x_array"])
        bed_fill = B2D((fill_y, fill_x))   # note swap
        surf_fill = S2D((fill_y, fill_x))  # note swap

    ax_layers.fill_between(fill_x, bed_fill, surf_fill, alpha=0.20, color = 'blue')
    ax_layers.fill_between(fill_x, np.nanmin(bed_fill) - 1500, bed_fill, alpha=0.35, color = 'gray')

    # Layer plot y-limits (keep your defaults, but safer)
    if Arguments.depth == "Elevation":
        ax_layers.set_ylim(-2000, 2000)

    # ax_layers.legend(loc="lower right", fontsize=8, framealpha=0.2)
    
    fs = 50
    
    ax_vel.text(0.01, 0.98, "Surface Speed $ma^{-1}$",
          ha="left", va="top", transform=ax_vel.transAxes, fontsize=fs)
    
    ax_strain.text(0.01, 0.98, f"Lateral Strain $a^{-1}$",
                ha="left", va="top", transform=ax_strain.transAxes, fontsize=fs)

    ax_tau.text(0.01, 0.98, "Basal Melt Rate $mm/year$",
                ha="left", va="top", transform=ax_tau.transAxes, fontsize=fs)
    
    ax_layers.text(0.01, 0.98, "Englacial Layering",
                    ha="left", va="top", transform=ax_layers.transAxes, fontsize=fs)

    # plt.tight_layout()
    plt.show()

    return bed_profile_interp, surf_profile_interp


# %%

class Arguments:
    
    limits = (-1.38e6,-1.275e6)
    
    # -1e5,-4e5 (Y67a)
    # -1.38e6,-1.275e6 (X47a)
    # -5.5e5,-5.2e5 (Y17a)
    # -5.5e5,-5.0e5 (Y19a)
    
    direction = 'x'
    depth = 'Elevation'
    upper_thresh = 60
    low_thresh = 10
    plot_style = 'scatter'
    point_size = 20
    strain_lims = [-0.05, 0.05]
    vel_lims = [0, 125]
    melt_mode = 'statistical'
    melt_lims = [10, 20]
    N = 5000
    G = .12
    
class Data:
    
    Vel = Vel
    all_layers = all_layers
    
class Files:
    surf_file = surface_file
    bed_file =  bed_file
    base_file = surf_file
    
class Interpolators:
    
    real_bed_interp = bm3["bed_interp"]
    real_surf_interp = bm3["surf_interp"]
    

Interpolators.bed_interp, Interpolators.surf_interp = frac_depth_plot(Files, Interpolators, Arguments, Data, Strains)

# %%

# folder = config.p("THW_Muldoon_Leuro_Layers") # folder containing horizon txt files

# # --- STORAGE ---
# all_pst_layers = {}  # dictionary of arrays keyed by filename

# for file in os.listdir(folder):
    
#     if not file.endswith(".csv"):
#         continue

#     filepath = os.path.join(folder, file)

#     # Read horizon file
#     df = pd.read_csv(filepath)

#     # Convert lat/lon -> projected coords
#     east, north = df['x'], df['y']

#     # Find Layer ELevation and Store EPS 1301 Locations
#     x_array = np.array(east)
#     y_array = np.array(north)
#     layer_depth = df["IRHdepth_m"].values
#     surface_elev = df['thk_m'] + df['bedelv_mABSL']
#     layer_elev = surface_elev - layer_depth

#     # Compute cumulative distance along track
#     dx = np.diff(x_array)
#     dy = np.diff(y_array)
#     segment_lengths = np.sqrt(dx**2 + dy**2)
#     x_position = np.concatenate([[0], np.cumsum(segment_lengths)])

#     all_pst_layers[file] = {
#         "x_array": x_array,
#         "y_array": y_array,
#         "z_array": layer_elev,
#         "x_position": x_position,
#     }

# %%


def estimate_layer_ages(all_layers,
                        Interpolators,
                        core_x,
                        core_y,
                        core_depths,
                        core_ages,
                        axis='y',
                        depth_is_positive_down=True):
    
    B = Interpolators.real_bed_interp
    S = Interpolators.real_surf_interp
    
    rel_B = Interpolators.bed_interp
    rel_S = Interpolators.surf_interp
    
    # prepare RGI interpolator mapping depth -> age
    core_depths = S((core_x,core_y)) - np.asarray(core_depths)
    core_ages = np.asarray(core_ages)

    # Ensure one-dimensional monotonic grid for RGI
    # RGI expects grid shape like (N,) and data shape (N,). Wrap depth in tuple.
    age_interp = RGI((core_depths,), core_ages, method='linear', bounds_error=False, fill_value = np.nan)   

    rows = []

    for old_name, layer in all_layers.items():

        idx = int(np.nanargmin(np.abs(layer['y_array'] - core_y)))
        x = np.array(layer['x_array'])[idx] 
        nearest_coord = np.array(layer['y_array'])[idx]
        frac_depth =  (rel_S([[nearest_coord]])[0] - np.array(layer['z_array'])[idx]) / (rel_S([[nearest_coord]])[0] - rel_B([[nearest_coord]])[0])
        layer_depth = S((x, nearest_coord)) - (frac_depth * (S((x, nearest_coord)) - B((x, nearest_coord))))
        age_est = age_interp([layer_depth])[0]
        new_name = f'{age_est:.2f}_ka_THW'
        
        # print(old_name, '\n')
        
        rows.append((old_name, new_name, nearest_coord, layer_depth, age_est))
        
        # print(rows,'\n')

    df = pd.DataFrame(rows, columns=['layer_name', 'layer age name','nearest_coord',
                                     'layer_depth', 'age_est']).set_index('layer_name')

    return df, age_interp

# %%

# all_pst_layers[file] = {
#     "x_array": x_array,
#     "y_array": y_array,
#     "z_array": layer_elev,
#     "x_position": x_position,
# }

# x = all_pst_layers['THW_LEURO_SHALLOW_2 - Copy (2).csv']['x_array']
# y = all_pst_layers['THW_LEURO_SHALLOW_2 - Copy (2).csv']['y_array']
# z = all_pst_layers['THW_LEURO_SHALLOW_2 - Copy (2).csv']['z_array']

# z[z < -10000] = np.nan

# plt.scatter(x,y,c=z,cmap='viridis', vmin = -500, vmax = 2000)
# plt.plot(-1.387e6, -5.60e5, 'w^', markersize = 18)
# plt.plot(-1.387e6, -5.602e5, 'r^', markersize = 12)

# plt.xlim(-1.6e6,-0.7e6)

# %%

age_table, age_interp = estimate_layer_ages(
    
    all_layers = all_layers,
    Interpolators = Interpolators,   # or S if available in your scope
    core_x = cu.Core_x,
    core_y = cu.Core_y,
    core_depths = cu.depth,            # depth grid from core upload (positive down)
    core_ages = cu.calibrated_age,     # calibrated ages corresponding to depths
    axis='y'                         # set to 'x' if you want to match on x instead
)

print(age_table.head())

# %%

# surf_vel = np.sqrt(Vel.fvx((base_x,base_y))**2 + Vel.fvy((base_x,base_y))**2)

# plt.style.use('dark_background') 

# # --- Create two stacked plots with shared x-axis ---
# fig, (ax_vel, ax_layers) = plt.subplots(
#     2, 1, figsize=(15,5), sharex=True,
#     gridspec_kw={'height_ratios': [1, 2]}
# )

# # --- Velocity plot (top) ---
# ax_vel.plot(base_x, surf_vel, 'y')
# ax_vel.set_ylabel("Velocity (m/yr)", color='y')
# ax_vel.tick_params(axis='y', labelcolor='y')

# for fname, layer in list(all_layers.items()):
    
#     x_pos = layer["x_array"]
#     rel_surf = surf_interp((x_pos))
#     rel_bed = bed_interp((x_pos))
#     # x_pos = layer["x_array"]
#     z_arr = (layer["z_array"] - rel_bed) / (rel_surf - rel_bed)
#     ax_layers.scatter(x_pos, z_arr, label=fname)
    
# --- Bed, surface, and layers plot (bottom) ---
# ax_layers.plot(base_x, Thickness / Thickness, 'w')
# ax_layers.plot(base_x, surface*0 , 'w')

# ax_layers.set_ylim(0, 1)
# ax_layers.fill_between(base_x, 1, y2=ax_layers.get_ylim()[0], color='gray', alpha=0.4)
# ax_layers.fill_between(base_x, 1, y2=0, color='blue', alpha=0.2)
# ax_layers.set_xlim(x_limits)
# # plt.gca().invert_yaxis() # Invert the y-axis
# # ax_layers.set_ylim(1, 0)

# ax_layers.set_xlabel("X (m)")
# ax_layers.set_ylabel("Elevation (m)")

# plt.tight_layout()
# plt.show()

# %%

from correct_layers import correct_layers_to_absolute_elevation

abs_elev_layers = correct_layers_to_absolute_elevation(all_layers,bm3["surf_interp"], bm3["bed_interp"])

# %% Find Shortening

shortening_results = lsf.shortening(abs_elev_layers, x_limits=None, elev_type="true_elevation", plot=True)

# %% Find Max Slope
import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
from layer_shape_functions import max_slope

slope_results = max_slope(abs_elev_layers, slope_method="moving_window",      # "moving_window" or "stride"
    window_dx=200.0, x_limits=(187500,192500))

# %% Find Fold Extrema

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
from layer_shape_functions import fold_extrema

extrema_results = fold_extrema(abs_elev_layers,show_height=True,                  
    show_half_wavelength=True, x_limits=(187500,192500))

# %%

x_limits = [50000, 100000]

# --- Create two stacked plots with shared x-axis ---
fig, (ax_vel, ax_layers) = plt.subplots(
    2, 1, figsize=(15,5), sharex=True,
    gridspec_kw={'height_ratios': [1, 2]}
)

# --- Velocity plot (top) ---
ax_vel.plot(base_x, surf_vel, 'y')
ax_vel.set_ylabel("Velocity (m/yr)", color='y')
ax_vel.tick_params(axis='y', labelcolor='y')

for fname, layer in list(all_layers.items())[2:]:
    
    rel_surf = bm3["surf_interp"]((layer["x_array"], layer["y_array"]))
    rel_bed = goff_bed_interp((layer["x_array"],layer["y_array"]))
    rel_thk = rel_surf - rel_bed
    x_pos = layer["x_array"]
    z_arr = (layer["z_array"]) / rel_thk
    ax_layers.plot(x_pos, z_arr, label=fname)
    
for fname, layer in slope_results.items():
    x_pt = layer['x_mid']
    z_pt = layer['z_mid']
    slope = layer['max_slope']

    # Scale arrow length (adjust factor as needed)
    vec_len = (x_limits[1] - x_limits[0]) * 0.5
    dx_vec = vec_len / np.sqrt(1 + slope**2)
    dz_vec = slope * dx_vec

    # Plot slope arrow
    ax_layers.quiver(
        x_pt, z_pt, dx_vec, dz_vec,
        angles='xy', scale_units='xy', scale=10,
        color='white', width=0.0025, headwidth=4, headlength=6
    )
    
# --- Bed, surface, and layers plot (bottom) ---
#ax_layers.plot(base_x, Thickness / Thickness, 'w')
ax_layers.plot(base_x, surface*0 , 'w')

# ax_layers.set_ylim(0, 1)
# ax_layers.fill_between(base_x, 1, y2=ax_layers.get_ylim()[0], color='gray', alpha=0.4)
# ax_layers.fill_between(base_x, 1, y2=0, color='blue', alpha=0.2)
# ax_layers.set_xlim(-1.411e6, -1.406e6)
plt.gca().invert_yaxis() # Invert the y-axis

ax_layers.set_xlabel("X (m)")
ax_layers.set_ylabel("Elevation (m)")

plt.tight_layout()
plt.show()

# %%
# s2yr = (60*60*24*365.25)

# surf_vel = np.sqrt(Vel.fvx((base_x,base_y))**2 + Vel.fvy((base_x,base_y))**2)

# theta = np.zeros(np.size(surf_vel))    # initialize

# upper_thresh = 55 
# low_thresh = 10

# # ramp section: between 30 and 85
# mask = (surf_vel > low_thresh) & (surf_vel < upper_thresh)
# theta[mask] = (surf_vel[mask] - low_thresh) / (upper_thresh - low_thresh)

# # upper bound
# theta[surf_vel >= upper_thresh] = 1

# m = 3; n = 3.0; rho = 917; L = 3.335e5; k = 2.1; dTdz = 0.04;
# tau_b = 1.2e5 *(((m*n + 1)/(2*Thickness))*(surf_vel)*(1 - theta))**(1/n); # basal shear stress
# melt_rate = (1/(L*rho))*((70/1e3) - (k*dTdz) + tau_b*((surf_vel*(1/s2yr))*theta)) # basal melt due to internal friction


# plt.plot(base_x,melt_rate*1000*s2yr)
# plt.plot(base_x, tau_b)

# %%

# import numpy as np
# from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# # Unstructured data points: (x, y, value)
# points = np.array([[2, 5], [0, 4], [1, 6], [2, 4], [0, 6], [1, 5]])
# values = np.array([10, 20, 30, 40, 50, 60])

# # Create the interpolator
# interp_func = LinearNDInterpolator(points, values)

# # Points to interpolate at
# points_to_interpolate = np.array([[1, 5], [0.5, 4.5], [1.5, 5.5]])

# # Perform the interpolation
# interpolated_values = interp_func(points_to_interpolate)
# print(interpolated_values)

# %%

# import numpy as np
# from scipy.interpolate import RegularGridInterpolator

# # Example non-monotonic data
# x_unsorted = np.array([2., 0., 1.])
# y_unsorted = np.array([5., 4., 6.])
# values_unsorted = np.array([[10., 40., 60.],
#                             [20., 50., 70.],
#                             [30., 60., 80.]]) # Example values array (3x3)

# # Sort the x and y coordinates and reorder values
# x_sorted_indices = np.argsort(x_unsorted)
# y_sorted_indices = np.argsort(y_unsorted)

# x_sorted = x_unsorted[x_sorted_indices]
# y_sorted = y_unsorted[y_sorted_indices]

# Reorder values using meshgrid-like indexing
# This step is complex for general cases but required for RGI
# For this specific 3x3 example, we would need to map the values correctly.
# A simpler approach for *unstructured* data is Option 2.
    
# %%
    
# palette = [
#     "#D8DBE2",
#     "#A9BCD0",
#     "#58A4B0",
#     "#373F51",
#     "#1B1B1E",
# ]

# plt.hist(n, bins = 100)

# i = 0

# fig, ax = plt.subplots(1, 1, figsize=(35,7))

# # Transparent background
# fig.patch.set_alpha(0)#0cd3f0
# ax.set_facecolor('none')

# while i < int(N/10):
#     ax.plot(base_x, melt_in_mm[i], alpha = 0.2, color = palette[2])
#     i += 10

# ax.set_ylim(-3,20)
# ax.set_xlim(-1.38e6,-1.285e6)
    
# mean = melt_in_mm.mean(axis=0)
# std = np.std(melt_in_mm, axis=0)
# p05  = np.percentile(melt_in_mm, 5, axis=0)
# p95  = np.percentile(melt_in_mm,95, axis=0)

# plt.fill_between(
#     base_x, mean + 2*std, mean - 2*std,
#     color=palette[1],
#     alpha=0.35,
#     label=r"2$\sigma$"
# )

# plt.plot(
#     base_x, mean,
#     color=palette[4],
#     linewidth=3.5,
#     label="Mean"
# )

# plt.grid(linestyle = '--', linewidth = 0.4, color = 'white')

# ax.tick_params(size = 20, labelsize = 20, pad = 10)

# plt.title('X47a Basal Melt Profile', size = 30, pad = 30)

# ax.set_xlabel("Eastings (EPS: 1301) $meters$", size = 25, labelpad = 30)
# ax.set_ylabel("Basal Melt (mm/year)", size = 20, labelpad = 20)

# # Adjust font, markers, and spacing
# ax.legend(loc='upper center', fontsize=15, markerscale=1.5,
#           labelspacing=0.8, handlelength=2.5, borderpad=2, facecolor=palette[3])

# plt.show()




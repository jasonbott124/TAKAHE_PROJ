# -*- coding: utf-8 -*-

"""
Flowline Tracing .py

Created on Thu Apr 17 13:17:41 2025

@author: Jason
"""

# %% Imports
import config  # central data/output paths (config.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import interpolate
from pyproj import Proj, transform
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import xarray as xr
import sys
pass  # (was a hard-coded sys.path entry; repo modules import directly)
from Flowband_Functions import *
from scipy.interpolate import RegularGridInterpolator as RGI
import cmocean

# %% File Paths

Mv2 = nc.Dataset(config.MV2_VEL)

CD_GHfx = config.p(r"Modeling Data (Sticky Spot)/Geothermal Heat Flux/200km_GHF_windowCenters.val.tiff")
Seis_GHfx = config.p(r"Modeling Data (Sticky Spot)/Geothermal Heat Flux/GeothermalHeatFlux_5km.tif")
ALBMAP_GHfx_Fox = config.p(r"Modeling Data (Sticky Spot)/ALBMAP/ALBMAP_GeothermalHeatFlux_FoxMaule_5km.tif")
ALBMAP_GHfx_Shapiro = config.p(r"Modeling Data (Sticky Spot)/ALBMAP/ALBMAP_GeothermalHeatFlux_ShapiroRitzwoller_5km.tif")
ALBMAP_SurfTemp = config.p(r"Modeling Data (Sticky Spot)/ALBMAP/ALBMAP_SurfaceAirTemperature_5km.tif")
RACMO_SurfTemp = config.p(r"Modeling Data (Sticky Spot)/Van Wessem RACMO/RACMO_Temperature_2m_35km.tif")

RACMO_Accumulation = config.p(r"Modeling Data (Sticky Spot)/Accumulation/precip.KNMI-2011.XPEN055.ERAINx_RACMO2.4.3H.nc")
RACMO_Precipitation = config.p(r"Modeling Data (Sticky Spot)/Van Wessem RACMO/RACMO_Precipitation_35km.tif")

RACMO_Sublimation = config.p(r"Modeling Data (Sticky Spot)/Van Wessem RACMO/RACMO_Sublimation_Rate_35km.tif")

Basal_Melt_5km = config.p(r"Modeling Data (Sticky Spot)/Ice Melt & Friction/waismelt.5km.grd")

SAMBA_SMBalance = config.p(r"Modeling Data (Sticky Spot)/SAMBA Surface mass balance measurements/SAMBA_surface_mass_balance_measurements.shp")
RACMO_SMBalance = config.p(r"Modeling Data (Sticky Spot)/Van Wessem RACMO/RACMO_SurfaceMassBalance_35km.tif")

flt_surf = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_surface.flt")
hdr_surf = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_surface.hdr")

flt_bed = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_bed.flt")
hdr_bed = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_bed.hdr")

flt_thk = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_thickness.flt")
hdr_thk = config.p(r"bedmap2_bin/bedmap2_bin/bedmap2_thickness.hdr")

Johns_bed_path = config.p(r"Modeling Data (Sticky Spot)/Beds/John Goff's Conditional Simulation/thwaitsbed_chan_consim4.grd")

bm3_bed_path = config.BM3_BED
bm3_surf_path = config.BM3_SURF

bmach_bed_path = config.p(r"BedMachinev3_Usable.tif")

# %% Define Beds from bm3, Goff et al., bm2

class bm3:

    bed = new_tif_to_array(bm3_bed_path)
    surf = new_tif_to_array(bm3_surf_path)

    bed_sub = bed[7200:8300,3200:4300].astype(float)
    surf_sub = surf[7200:8300,3200:4300].astype(float)
    surf_sub[surf_sub < -5000] = np.nan

    xc = np.arange(-3333500,3333500,500)
    yc = np.arange(3333500,-3333500,-500)

    [Xc, Yc] = np.meshgrid(xc,yc)
    
    x_sub = xc[3200:4300]
    y_sub = yc[7200:8300]
    
    [X_sub,Y_sub] = np.meshgrid(x_sub,y_sub)
    

# class BedMachine:
    
#     bed = new_tif_to_array(bmach_bed_path)

#     bed_sub = bed[7200:8300,3200:4300].astype(float)
#     # surf_sub[surf_sub < -5000] = np.nan

#     xc = np.arange(-3333250,3333250,500)
#     yc = np.arange(3333250,-3333250,-500)

#     [Xc, Yc] = np.meshgrid(xc,yc)
    
#     x_sub = xc[3200:4300]
#     y_sub = yc[7200:8300]
    
#     [X_sub,Y_sub] = np.meshgrid(x_sub,y_sub)
    

# plt.imshow(bm3.Xc,bm3.Yc,bm3.sub)

# %%

# Load the .grd file
data = xr.open_dataset(Johns_bed_path)

# Access the variable
print(data)  # This will show the structure of the dataset
Z = data['z']  # Replace 'variable_name' with the actual name of the data variable
x_data = data['x']
y_data = data['y']

# Convert to numpy array if needed
Elev = Z.values
goff_x = x_data.values
goff_y = y_data.values

[goff_X,goff_Y] = np.meshgrid(goff_x,goff_y)

goff_interp = RGI((goff_x, goff_y), Elev.T, method='cubic', bounds_error=False)

# %%

surf_raster_data, surf_hdr_data, srf_ncols, srf_nrows, srf_x, srf_y = binary_reader(hdr_surf,flt_surf)

bed_raster_data, bed_hdr_data, bed_ncols, bed_nrows, bed_x, bed_y = binary_reader(hdr_bed,flt_bed)

# thk_raster_data, thk_hdr_data, thk_ncols, thk_nrows, thk_x, thk_y = binary_reader(hdr_thk,flt_thk)

# %% Smooth Gflux Datasets

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv(config.p(r"Misc Data Files/Flowline_Data_StSp_Pages(Main(Altered)_with_Flow_Band_Width_FINAL).csv"))

GfluxCD = df['Geothermal Flux (CD)']
GfluxShp = df['Geothermal Flux (Shapiro)']
GfluxFox = df['Geothermal Flux (Fox)']
total_distance = df['totaldistance']/1000

X = df['X Coordinates']
Y = df['Y Coordinates']

smoothed_GfluxCD = gaussian_filter1d(GfluxCD, sigma=12)
smoothed_GfluxShp = gaussian_filter1d(GfluxShp, sigma=12)
smoothed_GfluxFox = gaussian_filter1d(GfluxFox, sigma=12)

# plt.style.use('dark_background')  # sets white lines/text on transparent background
# fig, ax = plt.subplots()

# # Plot the data
# ax.plot(total_distance, smoothed_GfluxCD, color='blue',linewidth=2.5,label='Airborne Magnetic Inversion')
# ax.plot(total_distance, smoothed_GfluxShp, color='red',linewidth=2.5, label = 'Seismic Inversion')
# ax.plot(total_distance, smoothed_GfluxFox, color='yellow',linewidth=2.5, label = 'Satelite Magnetic Inversion')
# plt.legend()

# Labels and title
# ax.set_ylabel('Geothermal Flux [$mW/m^2$]', color='white')
# ax.set_xlabel("Flowband's Distance From Grounding Line [km]", color='white')
# ax.set_title('Basal Melt Inversion [Joughin et al. 2009]', color='white')

# # White grid
# ax.grid(True, color='white', alpha=0.3)

# # White tick labels
# ax.tick_params(colors='white')

# # Transparent background
# fig.patch.set_alpha(0)
# ax.set_facecolor('none')

# plt.show()

# # plt.savefig('melt_plot.png', transparent=False, dpi=300)

# plt.plot(X[100:500],Y[100:500])

# %% Define Extents and Plot Region

# bm2_sub = surf_raster_data[3700:4100, 1750:2200]  # Proper 2D slicing
# bm2_bed = bed_raster_data[3700:4100, 1750:2200]  
# thk_sub = thk_raster_data[3700:4100, 1750:2200]

extents_x = [3500,4200]
extents_y = [7300,8100]

x_sub = bm3.xc[extents_x[0]:extents_x[1]]
y_sub = bm3.yc[extents_y[0]:extents_y[1]]
bm3.bed_sub = bm3.bed[extents_y[0]:extents_y[1],extents_x[0]:extents_x[1]]
bm3.surf_sub = bm3.surf[extents_y[0]:extents_y[1],extents_x[0]:extents_x[1]]

surf_x_subset,surf_y_subset = srf_x[3700:4100], -srf_y[1750:2200]

X_sub,Y_sub = np.meshgrid(surf_x_subset,surf_y_subset)

#y_idx = np.where((y<=surf_y_subset.max()) & (y>= surf_y_subset.min()))[0]
#x_idx = np.where((x<=surf_x_subset.max()) & (x>=surf_x_subset.min()))[0]

# print(np.shape(Elev))
# bed = Elev[np.ix_(y_idx,x_idx)]
# x_sub = x[x_idx]
# y_sub = y[y_idx]

(bm3.X_sub,bm3.Y_sub) = np.meshgrid(x_sub,y_sub)

# Compute direction of arrows
# dx = np.gradient(X)
# dy = np.gradient(Y)
# magnitude = np.hypot(dx, dy)
# dx /= magnitude
# dy /= magnitude

# Create a figure with 2 subplots
figure, ax = plt.subplots(2, figsize=(20, 12))

# First subplot
im1 = ax[0].imshow(bm3.surf_sub, cmap='viridis', vmin=0, vmax=4000, 
                   extent=[x_sub.min(), x_sub.max(), 
                           y_sub.min(), y_sub.max()])
# ax[0].plot(X[200:550],Y[200:550])
# Plot dashed line
# for i in range(0, len(X), 50):  # every few steps
#     ax[0].annotate('', xy=(X[i+1], Y[i+1]), xytext=(X[i], Y[i]),
#                  arrowprops=dict(arrowstyle='<-', color='white'))
# ax[0].plot(X[200:550], Y[200:550], linestyle='--', color='white')
# Add arrows
# ax[0].quiver(X[200:550], Y[200:550], dx, dy, scale=50, width=0.03, color='white')
figure.colorbar(im1, ax=ax[0], label='Surface Elevation (meters)')
ax[0].set_title('Map of Surface (Western Thwaites)')
ax[0].set_xlabel('Eastings (m)')
ax[0].set_ylabel('Northings (m)')

# Second subplot
im2 = ax[1].imshow(bm3.bed_sub, extent=[bm3.X_sub.min(), bm3.X_sub.max(), 
        bm3.Y_sub.min(), bm3.Y_sub.max()], 
                   origin='upper', cmap='viridis')
# ax[1].plot(X[200:550],Y[200:550])
figure.colorbar(im2, ax=ax[1], label='Elevation [m]')
ax[1].set_xlabel('Northings [m]')
ax[1].set_ylabel('Eastings [m]')
ax[1].set_title('Map of Western Thwaites Glacier Bed Elevation')
ax[1].grid(color='k', linestyle='--', linewidth=0.4)

# Display the plot
plt.tight_layout()  # Adjust layout to prevent overlap

# %% Plot Surface in 3D

bm3.surf_sub[np.where(bm3.surf_sub < -999)] = 0

fig = plot_Surface_3D(bm3.surf_sub,bm3.X_sub,bm3.Y_sub,bm3.x_sub,bm3.y_sub, DataType='Surface', plot2D=False, exag = 0.05, cmapv = bm3.surf_sub)
fig.write_html("bmchv3_surf.html", full_html=True, include_plotlyjs='cdn')
plt.show()

# fig = plot_Surface_3D(bm3.surf_sub,bm3.X_sub,bm3.Y_sub,x_sub,y_sub,DataType='Surface',plot2D=False, exag = 0.04)
# fig.write_html("bm3_surf.html", full_html=True, include_plotlyjs='cdn')
# plt.show()

# %% Plot Bed in 3D

# fig = plot_Surface_3D(bm3.bed_sub,bm3.X_sub,bm3.Y_sub,x_sub,y_sub,DataType='Bed',plot2D=False, exag = 0.05, cmapv = bm3.bed_sub)
fig = plot_Surface_3D(goff_interp((bm3.X_sub,bm3.Y_sub)),goff_X,goff_Y,goff_x,goff_y,DataType='Bed',plot2D=False, exag = 0.05, cmapv = Elev)
fig.write_html("goff_bed.html", full_html=True, include_plotlyjs='cdn')

# %% Define Velocity Data

ylim = [np.min(-X_sub), np.max(-X_sub)]  # X-coordinate limits
xlim = [np.min(-Y_sub), np.max(-Y_sub)]   # Y-coordinate limits

x = Mv2['x'][:].data
y = Mv2['y'][:].data

xindex = np.where((x > xlim[0]) & (x < xlim[1]))
yindex = np.where((y > ylim[0]) & (y < ylim[1]))

x_filtered = x[xindex]
y_filtered = y[yindex]

class Vel:
    
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

velocity_magnitude = np.sqrt(vx_filtered**2 + vy_filtered**2)

bm3.bed_sub[bm3.bed_sub == np.nan] = 0 # maybe?

# 1. Create a RegularGridInterpolator for sub
bm3.interpolator = RGI((bm3.X_sub[0, :], bm3.Y_sub[:, 0]), bm3.surf_sub.T, method='linear', bounds_error=False, fill_value=None)

# 2. Interpolate the 'sub' surface onto the velocity grid (X, Y)
bm3_interpolated = bm3.interpolator((Vel.Xc, Vel.Yc))

fig = plot_Velocity(bm3_interpolated,velocity_magnitude,X=Vel.Xc,Y=Vel.Yc,v_exag=0.09,cmocean_map=cmocean.cm.haline,plot2D=False,contours=False)

fig.write_html("bm3_surf_with_velocity.html", full_html=True, include_plotlyjs='cdn')

# plt.figure(figsize=(6, 5))
# ax = plt.gca()
# im1 = ax.pcolormesh(Vel.Xc, Vel.Yc, bm3_interpolated, cmap='gnuplot', shading='auto', vmin=-2000, vmax=2000)

# %% Plot Flowband in 3D

xls = pd.ExcelFile(config.p(r"Flowband Code/Plotting_Coordinates_Flowband_Main_(Correct_With_Surface).xlsx"))
df1 = pd.read_excel(xls, 'Profile_1')
df2 = pd.read_excel(xls, 'Profile_2')
df3 = pd.read_excel(xls,'Profile_3')

k = 2500

x_main = df3['X Coordinates'][:k]
y_main = df3['Y Coordinates'][:k]
z_main = df3['Surface'][:k]
ss_main = df3['SS'][:k]

x_R = df2['X Coordinates'][:k]
y_R = df2['Y Coordinates'][:k]
z_R = df2['Surface'][:k]
ss_R = df2['SS'][:k]

x_L = df1['X Coordinates'][:k]
y_L = df1['Y Coordinates'][:k]
z_L = df1['Surface'][:k]
ss_L = df1['SS'][:k]

xlist = [x_main,x_R,x_L]
ylist = [y_main,y_R,y_L]
zlist = [z_main,z_R,z_L]
ss_list = [ss_main,ss_R,ss_L]

import plotly.io as pio
pio.renderers.default = 'browser'

i = 0

while i < 3:
    # Add the flowline as a 3D scatter trace with color based on ss
    fig.add_trace(go.Scatter3d(
        x=xlist[i],  # Flowline X coordinates
        y=ylist[i],  # Flowline Y coordinates
        z=zlist[i]+15,  # Flowline Z coordinates (e.g., elevation)
        mode='lines',
        line=dict(
            width=4,
            color=ss_list[i],  # Colors line based on surface slope
            colorscale=cmocean_to_plotly(cmocean.cm.solar),
            cmin=0.02,  # Normalizes colorscale to surface slope array
            cmax=0.004  
        ),
        # name=f'Flowline Realization {k,i+1}' # sets name of flowline realization
    ))
    
    print()
    
    i+=1
    
# ------------------------------

from matplotlib.path import Path
from scipy.interpolate import griddata

# Create grid between left and right flowlines
x_min, x_max = min(min(x_L), min(x_R)), max(max(x_L), max(x_R))
y_min, y_max = min(min(y_L), min(y_R)), max(max(y_L), max(y_R))

grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, 1000),
    np.linspace(y_min, y_max, 1000)
)

# Create polygon from flowlines (left and right)
polygon = np.vstack((np.column_stack((x_L, y_L)),
                     np.column_stack((x_R[::-1], y_R[::-1]))))
path = Path(polygon)

# Check which grid points are inside the polygon
grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
mask = path.contains_points(grid_points)
mask_reshaped = mask.reshape(grid_x.shape)

# Interpolate z and ss values onto the grid
points_all = np.vstack((np.concatenate([x_L, x_R]), 
                        np.concatenate([y_L, y_R]))).T
z_all = np.concatenate([z_L, z_R])
ss_all = np.concatenate([ss_L, ss_R])

grid_z = griddata(points_all, z_all, (grid_x, grid_y), method='linear')
grid_ss = griddata(points_all, ss_all, (grid_x, grid_y), method='linear')

# Mask outside the band
grid_z[~mask_reshaped] = np.nan
grid_ss[~mask_reshaped] = np.nan

# Plot the band as a surface
fig.add_trace(go.Surface(
    x=grid_x,
    y=grid_y,
    z=grid_z + 100,
    surfacecolor=grid_ss,
    colorscale=cmocean_to_plotly(cmocean.cm.solar),
    cmin=np.nanmin(ss_all),
    cmax=np.nanmax(ss_all),
    opacity=0.7,
    showscale=False
))


fig.write_html("bm3_surf_with_velocity.html", full_html=True, include_plotlyjs='cdn')
fig.show()
    

# %% Plot Flowband in 2D

shrink = 0.9

e1 = 200
e2 = 540
e3 = 110
e4 = 560

plt.figure(figsize=(20, 10))
ax = plt.gca()
im5 = plt.imshow(velocity_magnitude[e1:e2,e3:e4],cmocean.cm.haline,origin='upper', vmin=0,vmax=1100,
                extent=(Vel.Xc[e1:e2,e3:e4].min(),Vel.Xc[e1:e2,e3:e4].max(),Vel.Yc[e1:e2,e3:e4].min(),Vel.Yc[e1:e2,e3:e4].max()))
# contour_levels = np.linspace(100, 120, 5)  # Define contour levels
# contours = plt.contour(Xc, Yc, interp_CD((Xc,Yc)), levels=contour_levels, colors='black', linewidths=0.8)
plt.colorbar(im5, ax=ax, label='$m/year$',shrink=shrink)
# plt.clabel(contours, inline=True, fontsize=8) # , fmt='%1.0f')
ax.set_xticks([-1.5e6, -1.4e6, -1.3e6])
ax.set_yticks([-5e5, -5.75e5, -6.5e5])
plt.plot(x_R,y_R,color='black')
plt.plot(x_L,y_L,color='black')

# # Fill between flowlines (band)
# plt.fill(np.concatenate([x_L, x_R[::-1]]),
#          np.concatenate([y_L, y_R[::-1]]),
#          color='viridis', alpha=0.5, label='Flowband')

# Plot filled band colored by ss using pcolormesh
ss_plot = ax.pcolormesh(grid_x, grid_y, grid_ss, cmap=cmocean.cm.solar, alpha=0.8)
plt.colorbar(ss_plot, ax=ax, label='SS Value', shrink=0.9)

# %% Define interpolators for bm3 bed and surface

interp_bm3_bed = build_interpolator(bm3.Xc,bm3.Yc,bm3.bed)
interp_bm3_surf = build_interpolator(bm3.Xc,bm3.Yc,bm3.surf)

# %% Import pre-existing flowline coordinates

# Load data
df = pd.read_csv(config.p(r"Misc Data Files/Flowline_Data_StSp_Pages(Main(Altered)_with_Flow_Band_Width_FINAL).csv"))
df1 = pd.read_csv(config.p(r"Flowband Code/Usable_HandDrawn_Coordinates.csv"))

# Combine X and Y coordinates from both dataframes
#flowline_x = pd.concat([df['X Coordinates'], df1['X']], ignore_index=True)
#flowline_y = pd.concat([df['Y Coordinates'], df1['Y']], ignore_index=True)

flowline_x = df['X Coordinates']
flowline_y = df['Y Coordinates']

bm3_bed_flowline = -interp_bm3_bed((flowline_x,flowline_y))

plt.plot(df['total distance'],bm3_bed_flowline)
plt.plot(df['total distance'], df["John Goff's Bed"])

# plt.plot(flowline_x,flowline_y)

# plt.figure(figsize=(6, 5))
# ax = plt.gca()
# im4 = plt.imshow(interp_bm3_bed((bm3.X_sub,bm3.Y_sub)),'turbo',origin='upper', vmin=-2000,vmax=3000,
#                 extent=(bm3.X_sub.min(),bm3.X_sub.max(),bm3.Y_sub.min(),bm3.Y_sub.max()))
# contour_levels = np.linspace(-30, -10, 10)  # Define contour levels
# # contours = plt.contour(Xc, Yc, interp_Ts((Xc,Yc)), levels=contour_levels, colors='black', linewidths=0.8)
# plt.colorbar(im4, ax=ax, label='$W*m^{-2}$',shrink=shrink)
# # plt.clabel(contours, inline=True, fontsize=8) # , fmt='%1.0f')
# # ax.set_xticks([-1.5e6, -1.4e6, -1.3e6])
# # ax.set_yticks([-5e5, -5.75e5, -6.5e5])
# plt.scatter(flowline_x,flowline_y,color='black')

# %% Histograms of the Bed Datasets

import matplotlib.pyplot as plt
import numpy as np

# Flatten arrays and remove NaNs
bm3_bed_flat = bm3.bed_sub.flatten()
bm3_bed_flat = bm3_bed_flat[~np.isnan(bm3_bed_flat)]

bmach_bed_flat = BedMachine.bed_sub.flatten()
bmach_bed_flat = bmach_bed_flat[~np.isnan(bmach_bed_flat)]

john_bed_flat = Elev.flatten()
john_bed_flat = john_bed_flat[~np.isnan(john_bed_flat)]

plt.style.use('dark_background')
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Helper function to scale histogram
def plot_scaled_hist(ax, data, bins, color, title):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.bar(bin_centers, counts / len(data), width=bin_edges[1]-bin_edges[0],
           color=color, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel('Relative Frequency')
    ax.set_xlim(-3000, 2000)

# Plot histograms
plot_scaled_hist(axs[2], bm3_bed_flat, bins=100, color='blue', title='Bedmap3 Bed Elevation')
plot_scaled_hist(axs[1], bmach_bed_flat, bins=100, color='pink', title='BedMachine Bed Elevation')
plot_scaled_hist(axs[0], john_bed_flat, bins=100, color='orange', title='John Goff Bed Elevation')
axs[2].set_xlabel('Bed Elevation (m)')

# Transparent background
fig.patch.set_alpha(0)
for a in axs:
    a.set_facecolor('none')

plt.tight_layout()
plt.show()

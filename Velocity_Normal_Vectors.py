# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:09:34 2025

@author: Jason
"""
import config  # central data/output paths (config.py)
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import netCDF4 as nc
import math

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff

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
    
    bed_interp = RGI(
    (y_sub, x_sub),       # Axis order is (Y, X)
    bed_sub.T,            # Must match shape (len(Y), len(X))
    method='linear',      # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
    [X_sub,Y_sub] = np.meshgrid(x_sub,y_sub)

# %%
# --- Your velocity dataset setup ---
# assumes x_filtered, y_filtered, Mv2, xindex, yindex are already defined

Mv2 = nc.Dataset(config.MV2_VEL)

ylim = [np.min(bm3.X_sub), np.max(bm3.X_sub)]  # X-coordinate limits
xlim = [np.min(bm3.Y_sub), np.max(bm3.Y_sub)]   # Y-coordinate limits

x = Mv2['x'][:].data
y = Mv2['y'][:].data

xindex = np.where((x > xlim[0]) & (x < xlim[1]))
yindex = np.where((y > ylim[0]) & (y < ylim[1]))

x_filtered = x[xindex]
y_filtered = y[yindex]

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

# Optional perpendicular (normal) vectors, rotated +90°
nx = -uy
ny = ux

# %% PLOT A

# --- Interpolator for unit flow vectors ---
def unit_flow(xq, yq):
    """Return unit flow vector (ux,uy) at (xq,yq)."""
    u = fvx((xq, yq))
    v = fvy((xq, yq))
    spd = np.sqrt(u**2 + v**2)
    if spd < 1e-12 or np.isnan(spd):
        return np.nan, np.nan
    return u/spd, v/spd

# Example query
# uxq, uyq = unit_flow(10000, 25000)

# --- Visualization ---
Xc, Yc = np.meshgrid(x_filtered, y_filtered)

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 500    # e.g. 500 m/yr

plt.figure(figsize=(8,6))
plt.pcolormesh(Xc, Yc, speed, shading="auto", cmap="viridis",
               vmin=cbar_min, vmax=cbar_max)

plt.colorbar(label="Speed")
# plt.quiver(Xc[::10, ::10], Yc[::10, ::10],
#            ux[::10, ::10], uy[::10, ::10],
#            color="r", scale=40, width=0.003)
plt.title("Unit Flow Direction Vectors")
plt.gca().set_aspect("equal")
plt.title('Plot A')

# plt.ylim(-6e5, -5e5)
# plt.xlim(-1.5e6, -1.3e6)

plt.show()

#%%

import numpy as np
from scipy.sparse import spdiags, eye, kron, hstack

# Grid sizes
Nx, Ny = np.size(ux[:,1]), np.size(uy[1,:])  # example values
dx = np.mean(np.diff(x_filtered))   # grid spacing in x
dy = np.mean(np.diff(y_filtered))   # grid spacing in y

data = np.array([-np.ones(Nx),np.ones(Nx)]) / dx
diags = np.array([0, 1])
Dx = spdiags(data, diags, Nx, Nx+1).toarray()

data = np.array([-np.ones(Ny),np.ones(Ny)]) / dy
diags = np.array([0, 1])
Dy = spdiags(data, diags, Ny, Ny+1).toarray()

# --- 1D Identity matrices ---
Ix = eye(Nx, format="csr")
Iy = eye(Ny, format="csr")

# --- 2D tensor-product divergence matrices ---
Dx = kron(Iy, Dx, format="csr")  # 2D div-matrix in x
Dy = kron(Dy, Ix, format="csr")  # 2D div-matrix in y

# --- Complete 2D divergence operator ---
D = hstack([Dx, Dy], format="csr")

print("Dx shape:", Dx.shape)
print("Dy shape:", Dy.shape)
print("D shape :", D.shape)

G = -D.T

# %% PLOT B

# speed_flat = speed.ravel(order="F")

# du = (G*speed_flat).dot(nx)

# Stack components into last dimension
# vec = np.stack([nx, ny], axis=-1)  # shape: (1221, 1221, 2)

du = np.gradient(speed, dy, dx)

# Stack your normal vectors into an array of shape (2, Ny, Nx)
norm_vec = np.stack([nx, ny], axis=0)  # shape (2, 1221, 1221)

# Compute dot product along the first axis (components)
dir_du = np.sum(du * norm_vec, axis=0)  # shape: (1221, 1221)

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05    # e.g. 500 m/yr

plt.figure(figsize=(8,6))
plt.pcolormesh(Xc, Yc, dir_du, shading="auto", cmap="viridis", 
               vmin=cbar_min, vmax=cbar_max)
plt.title('PLOT B: Longitudinal Strain Rates')
bar = plt.colorbar()
# ax.label('$s^-1$')

# %% PLOT C: Flow Angle In Radians

# Quadrant III (x < 0, y < 0): Angle is \(180\degree +\theta _{ref}\) (or \(\pi +\theta _{ref}\) radians).
# Calculate Reference Angle: Use the formula \(\theta _{ref}=\arctan (\frac{y}{x})\).

normal_angles = np.arctan(ny/nx) + np.pi
plt.pcolormesh(Xc, Yc, normal_angles, shading="auto", cmap="turbo")
plt.title('PLOT C: Flow Angle (radians)')
bar = plt.colorbar()

# %% PLOT D: du/dx

dudx = np.gradient(vx_filtered, dy, dx)[0]
# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05  # e.g. 500 m/yr
plt.pcolormesh(Xc, Yc, dudx, shading="auto", cmap="viridis",
               vmin=cbar_min, vmax=cbar_max)
plt.title('PLOT D: du/dx')
bar = plt.colorbar()

# %% PLOT E: dv/dy

dvdy = np.gradient(vy_filtered, dy, dx)[1]
# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05  # e.g. 500 m/yr
plt.pcolormesh(Xc, Yc, dvdy, shading="auto", cmap="viridis",
               vmin=cbar_min, vmax=cbar_max)
plt.title('PLOT E: dv/dy')
bar = plt.colorbar()

# %% PLOT F: Devatoric Strain

dvdx = np.gradient(vy_filtered, dy, dx)[0]
dudy = np.gradient(vx_filtered, dy, dx)[1]

dev_strain = 0.5*(dvdx + dudy)

cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05  # e.g. 500 m/yr
plt.pcolormesh(Xc, Yc, dev_strain, shading="auto", cmap="viridis",
               vmin=cbar_min, vmax=cbar_max)
plt.title('PLOT F: Deviatoric Strain')
bar = plt.colorbar()

 # %% ROTATED COMPONENTS
 
a = normal_angles

xlims = [-1.7e6, -1.2e6]
ylims = [-7e5, -2e5]
 
lon_strain = dudx*(cos(a)**2) + 2*dev_strain*cos(a)*sin(a) + dvdy*(sin(a)**2)
lat_strain = dudx*(sin(a)**2) - 2*dev_strain*cos(a)*sin(a) + dvdy*(cos(a)**2)
shear_strain = (dvdy - dudx)*cos(a)*sin(a) + dev_strain*(cos(a)**2 - sin(a)**2)

# --- Plotting: 3 panels ---
fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

# Set your desired colorbar limits here:
cbar_min = -0.04     # e.g. 0 m/yr
cbar_max = 0.04   # e.g. 500 m/yr

n_ticks = 4

# Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
ax = axs[0]
pcm0 = ax.pcolormesh(Xc, Yc, lon_strain, shading='auto', cmap = 'seismic',
                     vmin=cbar_min, vmax=cbar_max)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_title('Longitudinal (Along-Flow) Strain Rate')
ax.set_aspect('equal')

fig.colorbar(pcm0, ax=ax, label='$s^{-1}$',
             orientation='horizontal', location='bottom', ticks = linspace(cbar_min,cbar_max,n_ticks),
             shrink=0.8, pad=0.02)

# Set your desired colorbar limits here:
# cbar_min = -0.05      # e.g. 0 m/yr
# cbar_max = 0.05    # e.g. 500 m/yr

# Panel 2: tangential velocity
ax = axs[1]
pcm1 = ax.pcolormesh(Xc, Yc, lat_strain, shading='auto', cmap = 'seismic',
               vmin=cbar_min, vmax=cbar_max)
ax.set_title('Lateral (Across Flow) Strain Rate')
ax.set_aspect('equal')

fig.colorbar(pcm1, ax=ax, label='$s^{-1}$',
             orientation='horizontal', location='bottom', ticks = linspace(cbar_min,cbar_max,n_ticks),
             shrink=0.8, pad=0.02)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Set your desired colorbar limits here:
# cbar_min = -0.05      # e.g. 0 m/yr
# cbar_max = 0.05    # e.g. 500 m/yr

# Panel 3: along-flow derivative map
ax = axs[2]
pcm2 = ax.pcolormesh(Xc, Yc, shear_strain, shading='auto', cmap = 'seismic',
               vmin=cbar_min, vmax=cbar_max)
ax.set_title('Shear Strain Rate')
ax.set_aspect('equal')

fig.colorbar(pcm2, ax=ax, label='$s^{-1}$',
             orientation='horizontal', location='bottom', ticks = linspace(cbar_min,cbar_max,n_ticks),
             shrink=0.8, pad=0.02)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

# %% PLOT H: EFFECTIVE SURFACE STRAIN RATE

ylims = [-6e5,-5e5]
xlims = [-1.5e6, -1.3e6]

eff_strain = 0.5*np.sqrt(dvdy**2 + dudx**2 + 2*(dev_strain)**2)

cbar_min = -0.1      # e.g. 0 m/yr
cbar_max = 0.1  # e.g. 500 m/yr
plt.pcolormesh(Xc, Yc, shear_strain, shading="auto", cmap="seismic",
               vmin=cbar_min, vmax=cbar_max)
plt.title('PLOT E: Effective Strain Rate')
bar = plt.colorbar()

plt.xlim(xlims)
plt.ylim(ylims)

# %% PACKAGE STRAIN MAPS

class Strain:
    lat_strain = lat_strain
    lon_strain = lon_strain
    eff_strain = eff_strain
    shear_strain = shear_strain
    
    lat_interp = RGI(
    (y_filtered, x_filtered),     # Axis order is (Y, X)
    lat_strain,                   # Must match shape (len(Y), len(X))
    method='linear',              # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
    lon_interp = RGI(
    (y_filtered, x_filtered),     # Axis order is (Y, X)
    lon_strain,                   # Must match shape (len(Y), len(X))
    method='linear',              # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
    eff_interp = RGI(
    (y_filtered, x_filtered),     # Axis order is (Y, X)
    eff_strain,                   # Must match shape (len(Y), len(X))
    method='linear',              # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
    shear_interp = RGI(
    (y_filtered, x_filtered),     # Axis order is (Y, X)
    shear_strain,                 # Must match shape (len(Y), len(X))
    method='linear',              # or 'cubic'
    bounds_error=False,
    fill_value=np.nan
    )
    
def Strain_Deliverer():
    return Strain

# %%

def thk_interp(X,Y):
    return bm3.surf_interp((X,Y)) - bm3.bed_interp((X,Y))

plt.pcolormesh(Xc, Yc, thk_interp(Xc,Yc), shading='auto', cmap = 'turbo',
               vmin=0, vmax=3000)
# ax.set_title('Ice Thickness (Bedmap 3)')
# ax.set_aspect('equal')

# fig.colorbar(pcm2, ax=ax, label='$s^{-1}$',
#              orientation='horizontal', location='bottom', ticks = linspace(cbar_min,cbar_max,n_ticks),
#              shrink=0.8, pad=0.02)


# %%
 

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.03    # e.g. 500 m/yr

# nx, ny shape: (Ny, Nx)
# Unit normal vectors (perpendicular to flow)
n_perp1_x = -ny  # +90° rotation
n_perp1_y = nx

n_perp2_x = ny   # -90° rotation
n_perp2_y = -nx

# Stack the normal vector for broadcasting
n_perp = np.stack([n_perp1_x, n_perp1_y], axis=0)  # shape: (2, Ny, Nx)

# Elementwise dot product: du • n_perp
# du has shape (2, Ny, Nx)
du_across = np.sum(du * n_perp, axis=0)  # shape: (Ny, Nx)

plt.figure(figsize=(8,6))
plt.pcolormesh(Xc, Yc, du_across, shading="auto", cmap="viridis", 
               vmin=cbar_min, vmax=cbar_max)

# %%

xlims = (-1.55e6, -1.2e6)
ylims = (-6e5,-5.0e5)

# --- Plotting: 3 panels ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Panel 1: speed + quiver of unit vectors (or raw velocities if preferred)
ax = axs[0]
pcm0 = ax.pcolormesh(Xc, Yc, speed, shading='auto', cmap = 'turbo', vmin=0, vmax=800)

s = 25

ax.quiver(Xc[::s, ::s], Yc[::s, ::s],
          ux[::s, ::s], uy[::s, ::s],
          color="r", scale=30, width=0.003)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_title('Surface Ice Speed and Flow Vectors [Rignot et al. 2017]')
ax.set_aspect('equal')

fig.colorbar(pcm0, ax=ax, label='Surface Ice Speed (v) $ma^{-1}$',
             orientation='horizontal', location='bottom',
             shrink=0.8, pad=0.02)

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05    # e.g. 500 m/yr

# Panel 2: tangential velocity
ax = axs[1]
pcm1 = ax.pcolormesh(Xc, Yc, dir_du, shading='auto',
               vmin=cbar_min, vmax=cbar_max)
ax.set_title('Longitudinal (Along-Flow) Surface Velocity Gradient')
ax.set_aspect('equal')

fig.colorbar(pcm1, ax=ax, label='$dv/dx$',
             orientation='horizontal', location='bottom',
             shrink=0.8, pad=0.02)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Set your desired colorbar limits here:
cbar_min = 0      # e.g. 0 m/yr
cbar_max = 0.05    # e.g. 500 m/yr

# Panel 3: along-flow derivative map
ax = axs[2]
pcm2 = ax.pcolormesh(Xc, Yc, du_across, shading='auto', 
               vmin=cbar_min, vmax=cbar_max)
ax.set_title('Lateral (Across-Flow) Surface Velocity Gradient')
ax.set_aspect('equal')

fig.colorbar(pcm2, ax=ax, label='$dv/dy$ (units/length)',
             orientation='horizontal', location='bottom',
             shrink=0.8, pad=0.02)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Optional: tighten, add NaN masking overlay
for ax in axs:
    # Mask locations where data is NaN for visibility
    ax.scatter(Xc[np.isnan(speed)], Yc[np.isnan(speed)], s=2, color='k', alpha=0.4)
    
fig.suptitle("Eastern Shear Margin", fontsize=18, y=0.85)

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:01:27 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the .mat file
# mat_data = scipy.io.loadmat(config.p(r"JBott_2.5D_Model_Code/Output_20250419_test8_JohnBed_Bm3Surf_ALBVnD/output_20250419_test8_JohnBed_Bm3Surf_ALBVnD.mat"))  # Replace with your .mat file
# mat_data = scipy.io.loadmat(config.p(r"JBott_2.5D_Model_Code/Output_20250419_test9_BedMachineBed_BedMachineSurf_ALBVnD/output_20250419_test8_BedMachineBed_BedMachineSurf_ALBVnD.mat"))
mat_data = scipy.io.loadmat(config.p(r"JBott_2.5D_Model_Code/Output_20250419_test7_Bm2Bed_Bm2Surf_ALBVnD/output_20250419_test7_JohnBed_Bm3Surf_ALBVnD.mat"))

# mat_data = scipy.io.loadmat(config.p(r"JBott_2.5D_Model_Code/Output_20250418_test3_Bm3_BedandSurf_ALBVnD/output_20250418_test3_Bm3_BedandSurf_ALBVnD.mat"))
input_data = scipy.io.loadmat(config.p(r"JBott_2.5D_Model_Code/Data_prep/inputs.mat"))

# Filter out internal MATLAB metadata (those starting with '__')
variables = {key: val for key, val in mat_data.items() if not key.startswith('__')}

inputs = {key: val for key, val in input_data.items() if not key.startswith('__')}


# bed_use = inputs.get('bed_use')
# bed_use = bed_use - np.abs((max(bed_use)))

# Access variables as NumPy arrays
for key, array in variables.items():
    print(f"{key}: shape {array.shape}, dtype {array.dtype}")
    # Example: do something with each array
    # array = np.ravel(array)  # Flatten if needed
    
df = pd.read_csv(config.p(r"Flowband Code/Flowline_Input.csv"))

bm3_bed = df["Bedmap3Bed"]
john = df["John Goff's Bed"]
bm2_bed = df["Bedmap2 Bed"]
BedMachine_bed = df['bedmach_bed']

bed_use = bm2_bed

bm2_surf = df['Surface']
BedMachine_surf = df['bedmach_surface']
bm3_surf = df['Bedmap3Surface']

# Example: extract a specific variable
x_layer = variables.get('x_layer')/1000 # - np.abs((min(John_bed)))
z_layer = variables.get('z_layer') - np.abs((min(bed_use)))
age_vec = variables.get('Age_vec') # - np.abs((min(John_bed)))
S_data = variables.get('S_data') - np.abs((min(bed_use)))
S_P = variables.get('S_P') - np.abs((min(bed_use)))
x_nodes = variables.get('x_nodes')/1000 
W_P = variables.get('W_P')
b_dot_P = variables.get('b_dot_P')


plt.style.use('dark_background') 
fig, ax = plt.subplots(figsize=(16,8))
ax = plt.gca()
for i in range(0, np.size(x_layer[:,1]), 2):
    age_label = f"{float(age_vec[:,i])/1000} ka"
    ax.plot(x_layer[i,:], z_layer[i,:], label=age_label)
    
ax.plot(x_nodes.T,S_P[49,:])
ax.plot(x_nodes.T,S_data,'w--')
ax.plot(x_nodes.T,bed_use,'w')
plt.xlim(0,4e2)
plt.ylim(-3000,2.5e3)
plt.xlabel('Distance From Grounding Line (km)')
plt.ylabel('Elevation (m)')
ax.fill_between(x_nodes.T.flatten(), bed_use, y2=ax.get_ylim()[0], color='gray', alpha=0.4)
ax.fill_between(x_nodes.T.flatten(), bed_use, y2=S_P[1,:], color='blue', alpha=0.2)
plt.title('2.5D Thermomechanical Model with Bedmap2 Bed DEM')
# plt.grid(True)

# Transparent background
fig.patch.set_alpha(0)#0cd3f0
ax.set_facecolor('none')

# Stack the 3 bed profiles (ensure they're all NumPy arrays)
bed_stack = np.vstack((john, BedMachine_bed, bm3_bed))

# Compute mean along axis=0 (column-wise mean)
mean_bed = np.nanmean(bed_stack, axis=0)  # Use nanmean to ignore any NaNs

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16,8))
ax = plt.gca()

# Plot beds
ax.plot(x_nodes.T, bm3_bed, label='Bedmap3 Bed', color='blue',linewidth=1)   # Orange
ax.plot(x_nodes.T, john, label="Goff et al. 2014", color='orange',linewidth=1)
# ax.plot(x_nodes.T, bm2_bed, label='Bedmap2 Bed', linewidth=1)   # Blue
ax.plot(x_nodes.T, BedMachine_bed, label='BedMachine Bed', color='pink', linewidth=1)  # Green
ax.plot(x_nodes.T, mean_bed, label = 'Arithmetic Mean', color='yellow',linestyle='--')

# Plot surfaces (dashed lines)
# ax.plot(x_nodes.T, bm2_surf, label='Bedmap2 Surface', color='#1f77b4', linestyle='--', linewidth=2)
# ax.plot(x_nodes.T, BedMachine_surf, label='BedMachine Surface', color='#2ca02c', linestyle='--', linewidth=2)
ax.plot(x_nodes.T, bm3_surf, label='INSAR-based Surface', color='white', linestyle='-', linewidth=2)

plt.xlim(0,4e2)
plt.ylim(-3000,2.5e3)

ax.fill_between(x_nodes.T.flatten(), mean_bed, y2=ax.get_ylim()[0], color='gray', alpha=0.4)
ax.fill_between(x_nodes.T.flatten(), mean_bed, y2=bm3_surf, color='blue', alpha=0.2)

# Labels and limits
ax.set_xlabel('Distance from Grounding Line (km)')
ax.set_ylabel('Elevation (m)')

# Legend
ax.legend()

# Transparent figure background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.show()

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots side-by-side

# John Goff's Bed
ax[0].hist(john, bins=30, color='orange', edgecolor='black')
ax[0].set_title("John Goff's Bed")
ax[0].set_xlabel('Elevation (m)')
ax[0].set_ylabel('Frequency')

# BedMachine Bed
ax[1].hist(BedMachine_bed, bins=30, color='pink', edgecolor='black')  # Green
ax[1].set_title('BedMachine Bed')
ax[1].set_xlabel('Elevation (m)')

# Bedmap3 Bed
ax[2].hist(bm3_bed, bins=30, color='blue', edgecolor='black')  # Orange
ax[2].set_title('Bedmap3 Bed')
ax[2].set_xlabel('Elevation (m)')

# Set transparent background for all subplots
fig.patch.set_alpha(0)
for a in ax:
    a.set_facecolor('none')

plt.tight_layout()
plt.show()

from scipy.interpolate import interp1d
import numpy as np

# Original data
x_original = x_nodes.flatten()  # Original x-axis (1310 points)
bed_original = bed_use          # Original bed elevation (1310 points)

# Create new x-points for interpolation (5000 points)
x_new = np.linspace(x_original.min(), x_original.max(), 5000)

# Interpolate bed to the new x-axis
interp_func = interp1d(x_original, bed_original, kind='cubic')
bed_interp = interp_func(x_new)

# Compute gradient (derivative)
dbed_dx = np.gradient(bed_interp, x_new)  # Gradient w.r.t. x_new

# Plot the interpolated bed and its gradient
import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot bed
ax[0].plot(x_new, bed_interp, color='cyan', label='Interpolated Bed')
ax[0].set_ylabel('Bed Elevation (m)')
ax[0].legend()

# Plot gradient
ax[1].plot(x_new, dbed_dx, color='magenta', label='Gradient (dbed/dx)')
ax[1].set_xlabel('Distance (km)')
ax[1].set_ylabel('Gradient (m/km)')
ax[1].legend()

plt.tight_layout()
plt.show()

plt.style.use('dark_background')
fig, ax = plt.subplots( figsize=(18, 6))  # 3 subplots side-by-side

# === Derivative of z_layer ===
dz_dx = np.gradient(z_layer, axis=1)  # Derivative along the x-direction (columns)

# === Plot results ===
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 4))

# Plot derivatives of the first few isochrones for clarity
for i in range(0, z_layer.shape[0], 14):  # every 4th layer to avoid clutter
    ax.plot(x_layer[i, :], np.gradient(z_layer[i, :]) - dbed_dx , label=f'dz/dx layer {i}')

# Plot derivative of the bed
# ax.plot(x_nodes.T, dbed_dx, 'w--', linewidth=2, label='dbed/dx')

# Formatting
ax.set_xlabel('Distance from Grounding Line (km)')
ax.set_ylabel('Elevation Gradient (m/km)')
ax.set_title('Derivatives of Isochrones and Bed Topography')
ax.set_xlim(0, 4e2)
ax.legend()
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.show()

import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram of the bed gradient (dbed_dx)
ax.hist((dz_dx - dbed_dx).flatten(), bins=50, color='blue', edgecolor='black')

# Formatting
ax.set_xlabel('Gradient of Bed Elevation (m/km)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Bed Elevation Gradient')
# ax.grid(color='white', alpha=0.3)

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.tight_layout()
plt.show()

# ax.legend()

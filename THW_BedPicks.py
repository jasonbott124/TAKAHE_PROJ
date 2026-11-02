# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:50:18 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import numpy as np
import pandas as pd
from Flowband_Functions import *
import matplotlib.pyplot as plt

# %%

mask_path = config.p(r"RadarLine_Coordinates_THW.tif")
data_path = config.p(r"Bed_Elevation_Thwaites_ASE1THW.tif")

mask = new_tif_to_array(mask_path)
data = new_tif_to_array(data_path)
# Convert data to float if it's not already
data = data.astype(float)

x = np.linspace(-1640500,-920500,720)
y = np.linspace(-133500,-901500,768)

(X,Y) = np.meshgrid(x,y)

# Mask all non-observation points
data[mask == 0] = np.nan
X[mask == 0] = np.nan
Y[mask == 0] = np.nan

# === Get only valid points (where all are NOT nan) ===
valid_mask = ~np.isnan(data) & ~np.isnan(X) & ~np.isnan(Y)

# Extract non-NaN values
x_valid = X[valid_mask]
y_valid = Y[valid_mask]
data_valid = data[valid_mask]

# Combine into DataFrame
df = pd.DataFrame({'X': x_valid, 'Y': y_valid, 'Value': data_valid})

# Save to CSV
df.to_csv("valid_data_points(test).csv", index=False)

plt.imshow(data[100:400, 200:500])

plt.plot(y_valid[0:100],data_valid[0:100])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming df is already created from your script
# Load the saved CSV for safety (or use df directly)
# df = pd.read_csv("valid_data_points(test).csv")

# data_valid = df['Value'].dropna()

plt.style.use('dark_background')

# Create the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Compute histogram values
counts, bin_edges = np.histogram(data_valid, bins=100)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot relative frequency
ax.bar(bin_centers, counts / len(data_valid), width=bin_edges[1]-bin_edges[0],
       color='cyan', edgecolor='black', alpha=0.7)

# Labels and limits
ax.set_title('Histogram of Bed Elevation (ASE1THW)')
ax.set_xlabel('Bed Elevation (m)')
ax.set_ylabel('Relative Frequency')
ax.set_xlim(-3000, 2000)  # Adjust limits as needed

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.tight_layout()
plt.show()

# %%

import pandas as pd

# --- Read file ---
file = config.p(r"AGASEA_09102007.txt")

# Read raw lines
with open(file, "r") as f:
    lines = f.readlines()

records = []
for line in lines:
    parts = line.strip().split()
    # First 6 are numeric
    time = float(parts[0])
    lon = float(parts[1])
    lat = float(parts[2])
    x_proj = float(parts[3])
    y_proj = float(parts[4])
    elevation = float(parts[5])
    # The last two are text, but "line_id" may have multiple words
    line_id = " ".join(parts[6:-1])  # join everything except last
    flag = parts[-1]
    records.append([time, lon, lat, x_proj, y_proj, elevation, line_id, flag])

colnames = ["time", "lon", "lat", "x_proj", "y_proj", "elevation", "line_id", "flag"]
df = pd.DataFrame(records, columns=colnames)

# --- Filter for Y29a ---
df_y29a = df[df["line_id"].str.contains("THW SJB0 Y29a")].copy()
print(df_y29a.head())
# %%
y_proj = df_y29a['y_proj']
x_proj = df_y29a['x_proj']
elevation = df_y29a['elevation']

dx = np.diff(x_proj)
dy = np.diff(y_proj)
segment_lengths = np.sqrt(dx**2 + dy**2)
line_position = np.concatenate([[0], np.cumsum(segment_lengths)])

plt.plot(line_position,elevation)


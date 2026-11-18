# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:20:59 2025

@author: Jason
"""

# Figure for Melt Rate Along the Flowline


import config  # central data/output paths (config.py)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv(config.p(r"Flowline_Data_StSp_Pages(Main(Altered)_with_Flow_Band_Width_FINAL).csv"))

GfluxCD = df['Geothermal Flux (CD)']
GfluxShp = df['Geothermal Flux (Shapiro)']
GfluxFox = df['Geothermal Flux (Fox)']
total_distance = df['totaldistance']/1000

X = df['X Coordinates']
Y = df['Y Coordinates']

smoothed_GfluxCD = gaussian_filter1d(GfluxCD, sigma=12)
smoothed_GfluxShp = gaussian_filter1d(GfluxShp, sigma=12)
smoothed_GfluxFox = gaussian_filter1d(GfluxFox, sigma=12)

plt.style.use('dark_background')  # sets white lines/text on transparent background
fig, ax = plt.subplots()

# Plot the data
ax.plot(total_distance, smoothed_GfluxCD, color='blue',linewidth=2.5,label='Airborne Magnetic Inversion')
ax.plot(total_distance, smoothed_GfluxShp, color='red',linewidth=2.5, label = 'Seismic Inversion')
ax.plot(total_distance, smoothed_GfluxFox, color='yellow',linewidth=2.5, label = 'Satelite Magnetic Inversion')
plt.legend()

# Labels and title
ax.set_ylabel('Geothermal Flux [$mW/m^2$]', color='white')
ax.set_xlabel("Flowband's Distance From Grounding Line [km]", color='white')
# ax.set_title('Basal Melt Inversion [Joughin et al. 2009]', color='white')

# White grid
ax.grid(True, color='white', alpha=0.3)

# White tick labels
ax.tick_params(colors='white')

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.show()

# plt.savefig('melt_plot.png', transparent=False, dpi=300)

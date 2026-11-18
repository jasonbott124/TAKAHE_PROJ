import config  # central data/output paths (config.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -------------------------------
# Load data
# -------------------------------
class Takahe:
    data = pd.read_csv(config.p(r"Flowband Code/Takahe_Profile_Plot.csv"))
    x = data['distance (micron)']
    z = data['bm3_bed_Band 1']

class Rainier:
    data = pd.read_csv(config.p(r"Flowband Code/Rainier_Profile_Plot.csv"))
    x = data['distance (micron)']
    z = data['Rainier_terrain_model_Band 1']

class Kilimanjaro:
    data = pd.read_csv(config.p(r"Flowband Code/Kilimanjaro_Profile_Plot.csv"))
    x = data['distance (micron)']
    z = data['kilimanjaro_utm_Band 1']


# -------------------------------
# Helper for smoothing + prominence
# -------------------------------
def smooth(z, frac=0.05, polyorder=3):
    """Smooth a dataset using Savitzky-Golay."""
    N = len(z)
    window = int(N * frac)
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2)
    return savgol_filter(z, window, polyorder)

def prominence(z):
    return z - z.min()


# Compute smoothed prominence curves
Tz = prominence(smooth(Takahe.z))
Rz = prominence(smooth(Rainier.z))
Kz = prominence(smooth(Kilimanjaro.z))


# -------------------------------
# Shift Rainier to the right
# -------------------------------
rainier_shift = -7000
n = 7000
Rx = Rainier.x.max() - Rainier.x + rainier_shift
Rx = Rx[n:np.size(Rx)]
Rz = Rz[n:np.size(Rz)]




# -------------------------------
# Global axis limits
# -------------------------------
xmin = min(Takahe.x.min(), Kilimanjaro.x.min(), Rx.min())
xmax = max(Takahe.x.max(), Kilimanjaro.x.max(), Rx.max())
zmax = max(Tz.max(), Rz.max(), Kz.max())


# -------------------------------
# Plot
# -------------------------------
fig  = plt.figure(figsize=(25, 6))

colors = {
    "Takahe": "yellow",
    "Rainier": "green",
    "Kilimanjaro": "royalblue"
}

s = 6000

# Curves
plt.plot(Takahe.x, Tz, color=colors["Takahe"], linewidth=2)
plt.plot(Rx, Rz, color=colors["Rainier"], linewidth=2)
plt.plot(Kilimanjaro.x + s, Kz, color=colors["Kilimanjaro"], linewidth=2)

# Shading
plt.fill_between(Kilimanjaro.x + s, Kz, 0, color=colors["Kilimanjaro"], alpha=0.15)
plt.fill_between(Takahe.x, Tz, 0, color=colors["Takahe"], alpha=0.15)
plt.fill_between(Rx, Rz, 0, color=colors["Rainier"], alpha=0.15)

# Labels and title (larger text)
plt.xlabel("Horizontal Distance (meters)", fontsize=20)
plt.ylabel("Prominence (meters)", fontsize=20)
plt.title("Prominence Profiles: Takahe, Rainier, and Kilimanjaro", fontsize=26)

# Larger tick labels
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Remove legend
# plt.legend()

# Text on each mountain (positions chosen visually)
plt.text(Takahe.x.mean()*1.5, Tz.max()*0.1, "Mt. Takahe", color=colors["Takahe"], fontsize=20)
plt.text(Rx.mean()*0.75, Rz.max()*0.45, "Mt. Rainier", color=colors["Rainier"], fontsize=20)
plt.text(Kilimanjaro.x.mean()*1.57, Kz.max()*0.40, "Mt. Kilimanjaro", color=colors["Kilimanjaro"], fontsize=20)

plt.xlim(0, xmax + s)
plt.ylim(0, zmax + 400)

plt.grid(color = 'white', linestyle = '--', linewidth = 0.4)

ax = plt.gca()

# Transparent background
fig.patch.set_alpha(0)#0cd3f0
ax.set_facecolor('none')

plt.tight_layout()
plt.show()

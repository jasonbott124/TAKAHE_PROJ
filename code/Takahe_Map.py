# takahe_bm3_contour_maps.py
# Uses Jason's existing BM3 uploader (load_bedmap3) with minimal/no changes.

import config  # central data/output paths (config.py)
import os
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# ---- Import YOUR uploader (no changes inside it) ----
# Put your uploader in a file, e.g. bm3_uploader.py, that defines load_bedmap3
from bm3_upload import load_bedmap3


# =========================
# User config
# =========================
BM3_SURF_PATH = config.BM3_SURF
BM3_BED_PATH = config.BM3_BED

OUT_DIR = "./outputs_takahe"
os.makedirs(OUT_DIR, exist_ok=True)

# Mt. Takahe center (edit if you want)
CENTER_LAT = -75.30
CENTER_LON = -112.10

RADIUS_KM = 80.0  # crop radius around center

CONTOUR_INTERVAL_M = 200.0
BOLD_EVERY = 5           # every Nth contour thicker
LABEL_CONTOURS = True
LABEL_EVERY_N = 2        # label every Nth contour level

DPI = 300
TRANSPARENT_PNG = True


# =========================
# BedMachine grid helpers (match your uploader)
# =========================
def bm3_full_coords():
    """
    Matches your uploader's coordinate construction:
      xc = np.arange(-3333500, 3333500, 500)
      yc = np.arange( 3333500,-3333500,-500)

    In your load_bedmap3():
      y_sub is sliced from xc using [col0:col1]
      x_sub is sliced from yc using [row0:row1]

    So:
      rows -> x (from yc, descending)
      cols -> y (from xc, ascending)
    """
    y_full = np.arange(-3333500, 3333500, 500)   # this is xc in your code
    x_full = np.arange( 3333500,-3333500,-500)   # this is yc in your code
    return x_full, y_full


def latlon_to_epsg3031_xy(lat, lon):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def compute_sub_idxs_from_center(center_x, center_y, radius_km):
    """
    Returns sub_idxs = [row0,row1,col0,col1] consistent with your uploader.
    """
    x_full, y_full = bm3_full_coords()

    r = radius_km * 1000.0
    xmin, xmax = center_x - r, center_x + r
    ymin, ymax = center_y - r, center_y + r

    # x_full is descending; y_full is ascending
    x_lo, x_hi = min(xmin, xmax), max(xmin, xmax)
    y_lo, y_hi = min(ymin, ymax), max(ymin, ymax)

    row_inds = np.where((x_full >= x_lo) & (x_full <= x_hi))[0]
    col_inds = np.where((y_full >= y_lo) & (y_full <= y_hi))[0]

    if row_inds.size == 0 or col_inds.size == 0:
        raise ValueError(
            "Crop window is outside BM3 grid. Check center coords or radius."
        )

    row0 = int(row_inds.min())
    row1 = int(row_inds.max()) + 1
    col0 = int(col_inds.min())
    col1 = int(col_inds.max()) + 1

    return [row0, row1, col0, col1]


def make_levels(Z, interval_m):
    finite = Z[np.isfinite(Z)]
    vmin, vmax = float(finite.min()), float(finite.max())
    start = np.floor(vmin / interval_m) * interval_m
    stop  = np.ceil(vmax / interval_m) * interval_m
    if stop <= start:
        stop = start + interval_m
    return np.arange(start, stop + interval_m, interval_m)


def contour_linewidths(levels, bold_every):
    lw = np.full(levels.shape, 0.6, dtype=float)
    if bold_every and bold_every > 0:
        lw[::bold_every] = 1.2
    return lw


# =========================
# Plotting
# =========================
def plot_epsg3031_contours(X, Y, Z, outpath):
    levels = make_levels(Z, CONTOUR_INTERVAL_M)
    lw = contour_linewidths(levels, BOLD_EVERY)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0 if TRANSPARENT_PNG else 1.0)

    cs = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=lw)

    if LABEL_CONTOURS:
        label_levels = levels[::max(1, LABEL_EVERY_N)]
        ax.clabel(cs, levels=label_levels, inline=True, fontsize=8, fmt="%d")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)  [EPSG:3031]")
    ax.set_ylabel("y (m)  [EPSG:3031]")
    ax.set_title("Mt. Takahe — BM3 Bed Elevation Contours (EPSG:3031)")

    ax.grid(True, linewidth=0.4, alpha=0.35)

    plt.savefig(outpath, dpi=DPI, bbox_inches="tight", transparent=TRANSPARENT_PNG)
    plt.close(fig)


def plot_latlon_contours(X, Y, Z, outpath):
    # Convert grid to lon/lat
    tf = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    Lon, Lat = tf.transform(X, Y)

    levels = make_levels(Z, CONTOUR_INTERVAL_M)
    lw = contour_linewidths(levels, BOLD_EVERY)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0 if TRANSPARENT_PNG else 1.0)

    cs = ax.contour(Lon, Lat, Z, levels=levels, colors="black", linewidths=lw)

    if LABEL_CONTOURS:
        label_levels = levels[::max(1, LABEL_EVERY_N)]
        ax.clabel(cs, levels=label_levels, inline=True, fontsize=8, fmt="%d")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Mt. Takahe — BM3 Bed Elevation Contours (Lat/Lon)")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    plt.savefig(outpath, dpi=DPI, bbox_inches="tight", transparent=TRANSPARENT_PNG)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    cx, cy = latlon_to_epsg3031_xy(CENTER_LAT, CENTER_LON)
    sub_idxs = compute_sub_idxs_from_center(cx, cy, RADIUS_KM)

    bm3 = load_bedmap3(BM3_BED_PATH, BM3_SURF_PATH, sub_idxs)

    # Your uploader returns:
    #   bed_sub: (rows, cols)
    #   X_sub, Y_sub: meshgrid(x_sub, y_sub) => shapes (cols, rows) if default meshgrid indexing
    #
    # BUT you built X_sub, Y_sub as:
    #   X_sub, Y_sub = np.meshgrid(x_sub, y_sub)
    # where x_sub length = rows, y_sub length = cols, so:
    #   X_sub shape = (cols, rows)
    #   bed_sub shape = (rows, cols)
    #
    # So we transpose bed_sub so everything matches X_sub/Y_sub.
    X = bm3["X_sub"]
    Y = bm3["Y_sub"]
    Z = bm3["bed_sub"].T  # <-- critical for consistent contouring with your meshgrid

    epsg_out = os.path.join(OUT_DIR, "takahe_bed_contours_epsg3031.png")
    ll_out   = os.path.join(OUT_DIR, "takahe_bed_contours_latlon.png")

    plot_epsg3031_contours(X, Y, Z, epsg_out)
    plot_latlon_contours(X, Y, Z, ll_out)

    print("Saved:")
    print(" -", epsg_out)
    print(" -", ll_out)


if __name__ == "__main__":
    main()

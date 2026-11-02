# -*- coding: utf-8 -*-
"""
layer_extent_map.py — picked isochrones plotted in the MapFigureMaker "HORIZONS + TAKAHE" style
(grid of map panels, one per layer, dense viridis scatter, horizontal colorbar under each,
"Isochron age: ... +/- ... ka" titles, Takahe triangle, dashed grid), on a WHITE background
(black text, ticks and outlines).

Loading / depth correction / dating are reused from layer_extent_maps.py (same folder):
  hor_allpst CSVs -> load_horallpst -> correct_layers_to_radar_surface -> collect
Makes 4 figures: elevation and fractional depth, each for the Takahe flank and the full extent.
"""
import config  # central data/output paths (config.py)
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = config.REPO_DIR
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import layer_extent_maps as lem                       # loaders, ages, Bedmap3, correction settings

# =============================================================================
# CONFIG (paths and ages live in layer_extent_maps.py)
# =============================================================================
OUT_DIR = os.path.join(lem.OUT_DIR, "isochron_panels")
EXTENTS = {                                           # x0, x1, y0, y1 in EPSG:3031 metres
    "Takahe": ([-1.575e6, -1.25e6], [-6.5e5, -4.7e5]),      # MapFigureMaker flank window
    "full": ([-1.7e6, -0.9e6], [-8e5, -1.5e5]),             # out to WAIS Divide (MapFigureMaker used x1 = -0.5e6)
}
LAYERS = None                       # e.g. ["D1", "D2", "D7", "D4", "D6"]; None = every picked layer
SKIP_EMPTY = True                   # drop a panel when the layer has no picks inside the extent
CMAP = "viridis"
CLIM_PCT = (0, 100)                 # colour range per panel; (0, 100) = plain min/max, as in MapFigureMaker
N_TICKS = 3                         # colorbar ticks
MARKER_SIZE = 200                   # scatter dot area (pt^2); None = matplotlib default (36)
FIGSIZE, DPI = (16, 9), 200
FONT = 16                           # base font size; titles +2, ticks -2
TAKAHE_XY = (-1.387e6, -5.60e5)     # as in MapFigureMaker
SHOW_WAIS_DIVIDE = True             # gold star on the full-extent figures
SHOW_UNPICKED_LINES = False         # light grey radar coverage under each layer (MapFigureMaker had none)
# =============================================================================


def _title(lid, var=None):
    what = {"elev": "Elevation", "frac": "Fractional thickness"}.get(var, "")
    a = lem.AGE_KEY.get(lid)
    if a is None:
        age = lid
    elif lid in lem.AGE_ERR and lid not in lem.NOT_REDATED:
        age = f"{a / 1000:.3f} ± {lem.AGE_ERR[lid] / 1000:.3f} ka"
    else:
        age = f"{a / 1000:.2f} ka"
    return f"{age} · {what}" if what else age


def _nice_ticks(lo, hi, n):
    t = np.linspace(lo, hi, n)
    span = hi - lo
    dec = 0 if span >= 50 else (1 if span >= 5 else (2 if span >= 0.5 else 3))
    return np.round(t, dec)


def isochron_panels(data, cover, var, extent, outpath, order):
    (x0, x1), (y0, y1) = EXTENTS[extent]
    inb = lambda D: (D.x >= x0) & (D.x <= x1) & (D.y >= y0) & (D.y <= y1)
    panels = []
    for lid in order:
        d = data[lid]
        d = d[inb(d) & np.isfinite(d[var])]
        if len(d) or not SKIP_EMPTY:
            panels.append((lid, d))
    n = len(panels)
    if n == 0:
        print(f"  [{extent}] no picks in this extent")
        return None
    nrow = 1 if n <= 3 else 2
    ncol = int(np.ceil(n / nrow))
    cc = cover[inb(cover)] if SHOW_UNPICKED_LINES else cover.iloc[:0]

    plt.rcParams.update({"font.size": FONT, "text.color": "white", "axes.labelcolor": "white",
                         "xtick.color": "white", "ytick.color": "white", "axes.edgecolor": "white"})
    fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, constrained_layout=True, squeeze=False,
                            facecolor="none")
    wd = lem.LL2PS.transform(lem.WAIS_DIVIDE[1], lem.WAIS_DIVIDE[0])
    for k, ax in enumerate(axs.flat):
        if k >= n:
            ax.axis("off")
            continue
        lid, d = panels[k]
        ax.set_facecolor("none")
        if len(cc):
            ax.scatter(cc.x, cc.y, s=0.5, c="0.82", lw=0, rasterized=True, zorder=1)
        v = d[var].to_numpy()
        lo, hi = (np.nanpercentile(v, CLIM_PCT) if v.size else (0, 1))
        if hi <= lo:
            lo, hi = lo - 1, hi + 1
        kw = {} if MARKER_SIZE is None else {"s": MARKER_SIZE}
        pcm = ax.scatter(d.x, d.y, c=v, cmap=CMAP, vmin=lo, vmax=hi, rasterized=True, zorder=3, **kw)
        cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom",
                          ticks=_nice_ticks(lo, hi, N_TICKS), shrink=0.6, pad=0.02)
        cb.set_label("Elevation asl (m)" if var == "elev" else "Fractional depth (0 = surface, 1 = bed)",
                     fontsize=FONT - 2)
        cb.ax.tick_params(labelsize=FONT - 3, width=1.2, length=4, color="white", labelcolor="white")
        cb.outline.set_edgecolor("white")

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(_title(lid, var), pad=8, fontsize=FONT + 2, color="white")
        ax.grid(color="white", alpha=0.35, linestyle="--", linewidth=0.6, zorder=0)
        ax.tick_params(labelsize=FONT - 2, width=1.2, length=5, direction="in", top=True, right=True)
        ax.ticklabel_format(style="sci", scilimits=(6, 6), useMathText=True)
        for s in ax.spines.values():
            s.set_edgecolor("white"); s.set_linewidth(1.2)

        # Takahe: black-outlined red triangle (MapFigureMaker used white/red on a dark background)
        ax.plot(*TAKAHE_XY, "w^", markersize=18, zorder=6)
        ax.plot(TAKAHE_XY[0], TAKAHE_XY[1] - 0.002e6, "^", color="red", markersize=12, zorder=7)
        if SHOW_WAIS_DIVIDE and extent == "full" and x0 <= wd[0] <= x1 and y0 <= wd[1] <= y1:
            ax.plot(*wd, "*", ms=18, mfc="#FFC20E", mec="white", mew=1.2, zorder=7)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=DPI, transparent=True)
    plt.close(fig)
    print("saved", outpath)
    return outpath


def make_isochron_panels(Corrected_PST_list=None, data_cover=None, out_dir=OUT_DIR):
    data, cover = data_cover or lem.collect(Corrected_PST_list)
    order = sorted(data, key=lambda k: lem.AGE_KEY.get(k, 1e9))
    if LAYERS:
        order = [k for k in order if k in LAYERS]
    out = []
    for var, vname in (("elev", "elevation"), ("frac", "fracdepth")):
        for ext in EXTENTS:
            out.append(isochron_panels(data, cover, var, ext,
                                       os.path.join(out_dir, f"isochrons_{vname}_{ext}.png"), order))
    return out


if __name__ == "__main__":
    from correct_layers_to_radar_surface import correct_layers_to_radar_surface
    PST_list = lem.load_horallpst()
    bm3 = lem.bedmap3_interps(PST_list)
    Corrected = correct_layers_to_radar_surface(PST_list, bm3["surf_interp"], bm3["bed_interp"],
                                                m_per_unit=lem.M_PER_UNIT, firn_corr_m=lem.FIRN_CORR_M,
                                                firn_depth_m=lem.FIRN_DEPTH_M)
    make_isochron_panels(Corrected)

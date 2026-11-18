# -*- coding: utf-8 -*-
"""
layer_extent_maps.py — map-view figures of every picked isochrone, coloured by
  (1) fractional depth near Takahe      (2) elevation near Takahe
  (3) fractional depth to WAIS Divide   (4) elevation to WAIS Divide
One PNG per layer per set, sized for a big screen (16:9, large text, thick ticks).

Depth/elevation come from your existing pipeline:
  picks (Landmark export) -> correct_layers_to_radar_surface (radar surface pick as the
  zero, time->depth at 0.8425 m/unit, firn correction) -> Bedmap3 surface anchors elevation.
  frac_depth = depth below surface / radar ice thickness  (0 = surface, 1 = bed)
  z_array    = Bedmap3 surface - depth                     (m above sea level)

Run as a script (edit CONFIG), or from your console with data already loaded:
    import layer_extent_maps as lem
    lem.make_all(Corrected_PST_list)          # already corrected with correct_layers_to_radar_surface
"""
import config  # central data/output paths (config.py)
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from pyproj import Transformer

# =============================================================================
# CONFIG
# =============================================================================
PICK_ROOT = config.p(r"Picked_PSTs_20261014/horallpst")
COORD = "latlon"                     # pick-file columns: label, col1, col2, lon, lat, elev  ("polar-stereo" if x/y)
BED_TIF = config.BM3_BED                    # Bedmap3 bed (EPSG:3031)
SURF_TIF = config.BM3_SURF   # Bedmap3 surface (EPSG:3031) — check filename
OUT_DIR = config.out("Layer_Extent_Maps")
PST_ROOT = None                      # per-line folders (X47a, Y67a, ...) holding Surface_ picks; None = parent of PICK_ROOT
SURFACE_RAW_DEFAULT = 250.0          # raw radar-surface value where no per-line Surface pick is nearby (picks read 235-272)
SURF_MATCH_M = 200.0                 # use a real Surface pick if one is within this distance

# correct_layers_to_radar_surface settings (same defaults as before)
M_PER_UNIT, FIRN_CORR_M, FIRN_DEPTH_M = 0.8425, 10.0, 100.0

# Layer ages (years). D1, D2, D7, D4, D6 re-dated at Y67a; D3, D5 not yet re-dated.
AGE_KEY = {"D1": 592, "D2": 1185, "D7": 1778, "D4": 2224, "D6": 4260, "D3": 2100, "D5": 2760}
NOT_REDATED = {"D3", "D5"}
AGE_ERR = {"D1": 14, "D2": 25, "D7": 38, "D4": 45, "D6": 99}   # ± (years) from the Y67a dating

MAKE_SINGLE = True                    # one PNG per layer per set (4 folders)
MAKE_GRIDS = True                     # one multi-panel PNG per set (all layers side by side)
GRID_DOT_SPACING_KM = 0.0             # 0 = every pick (dense swaths); e.g. 2.0 to thin
GRID_DOT_SIZE = 130

TAKAHE = (-76.283, -112.083)          # (lat, lon)
WAIS_DIVIDE = (-79.467, -112.085)     # WDC06A core site (lat, lon)
NEAR_HALF_KM = 120.0                  # near-Takahe map half-width (km) around TAKAHE_CENTER_OFFSET
NEAR_CENTER_OFFSET_KM = (0.0, 0.0)    # shift the near-Takahe window (east, north km) if the flank lines sit off-centre
FULL_PAD_KM = 25.0                    # padding around all picks + WAIS Divide

CMAP_FRAC, CMAP_ELEV = "plasma", "viridis"
CLIM_PCT = (2, 98)                    # colour limits from these percentiles of the points shown (per layer)
CLIM_FIXED = {}                       # e.g. {"frac": (0.0, 0.6), "elev": (-500, 1800)} to use the same limits for every layer
BACKGROUND = "transparent"            # "transparent" (no background, white text), "dark" (black) or "light" (white)
LABEL_LINES = []
                                      # line names written on the near-Takahe maps; True = all, False/[] = none
LABEL_LINES_FULL = []   # fewer labels on the full-extent maps (they get crowded)
DOT_SPACING_KM = {"near": 0.0, "full": 0.0}   # keep one pick per this spacing so dots read as separate points; 0 = all
DOT_SIZE = {"near": 200, "full": 130}           # scatter marker area (pt^2)
FIGSIZE, DPI = (16, 9), 200           # 3200 x 1800 px
BASE_FONT = 22                        # everything scales from this
# =============================================================================

PST_RE = re.compile(r"(?<![A-Za-z0-9])([XY]\d{2,3}[a-z])(?![A-Za-z0-9])")
LL2PS = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


# ----------------------------------------------------------------------------- loading
def _pst_token(*strings):
    for s in strings:
        m = PST_RE.search(str(s))
        if m:
            return m.group(1)
    return None


def load_picks(root=PICK_ROOT, coord=COORD):
    """
    Reads every .txt under `root` (any folder layout) into the same structure as
    cl.produce_layer_df: PST_list[pst][filename] = {x_array, y_array, z_array, x_position}.
    Handles both one-file-per-PST-per-horizon and one-file-per-horizon-for-all-lines
    (rows are split by the PST name found in the label column).
    """
    files = sorted(f for ext in ("*.csv", "*.txt")
                   for f in glob.glob(os.path.join(root, "**", ext), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .csv or .txt pick files under {root}")
    PST_list = {}
    for f in files:
        name = os.path.basename(f)
        df = pd.read_csv(f, header=None)
        if df.shape[1] < 6:
            print(f"[skip] {name}: {df.shape[1]} columns (expected 6)")
            continue
        df = df.iloc[:, :6]
        df.columns = ["label", "col1", "col2", "a", "b", "elev"]
        for c in ("col1", "col2", "a", "b", "elev"):            # tolerate a header row / text cells
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["a", "b"])
        if coord == "latlon":
            df["x"], df["y"] = LL2PS.transform(df["a"].values, df["b"].values)
        else:
            df["x"], df["y"] = df["a"].astype(float), df["b"].astype(float)
        file_pst = _pst_token(name, os.path.basename(os.path.dirname(f)))
        # PST = X/Y line name in the label (X47a), else the label's last "_" token (F46T01a, DRP05a),
        # else a name in the filename/folder (one-file-per-PST exports)
        df["pst"] = [(_pst_token(l) or (l.split("_")[-1] if "_" in l else None) or file_pst)
                     for l in df["label"].astype(str)]
        for pst, g in df.groupby("pst", dropna=False):
            if pst is None or (isinstance(pst, float) and np.isnan(pst)):
                print(f"[skip] {name}: could not find a PST name (e.g. X47a) in labels/filename/folder")
                continue
            g = g.sort_values(["col1", "col2"]) if g["col1"].dtype.kind in "if" else g
            x, y = g["x"].to_numpy(float), g["y"].to_numpy(float)
            seg = np.hypot(np.diff(x), np.diff(y))
            PST_list.setdefault(pst, {})[name] = {
                "x_array": x, "y_array": y, "z_array": g["elev"].to_numpy(float),
                "x_position": np.concatenate([[0.0], np.cumsum(seg)]),
            }
    print(f"loaded {len(files)} files -> {len(PST_list)} lines: {', '.join(sorted(PST_list))}")
    return PST_list


def bedmap3_interps(PST_list, bed_tif=BED_TIF, surf_tif=SURF_TIF, pad_m=20e3):
    """RegularGridInterpolators for Bedmap3 surface and bed, cropped to the picks (+pad)."""
    import rasterio
    from rasterio.windows import from_bounds
    from scipy.interpolate import RegularGridInterpolator
    xs = np.concatenate([L["x_array"] for P in PST_list.values() for L in P.values()])
    ys = np.concatenate([L["y_array"] for P in PST_list.values() for L in P.values()])
    b = (np.nanmin(xs) - pad_m, np.nanmin(ys) - pad_m, np.nanmax(xs) + pad_m, np.nanmax(ys) + pad_m)
    out = {}
    for key, path in (("surf_interp", surf_tif), ("bed_interp", bed_tif)):
        with rasterio.open(path) as src:
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            z = src.read(1, window=win).astype(float)
            if src.nodata is not None:
                z[z == src.nodata] = np.nan
            t = src.window_transform(win)
        z[z < -9000] = np.nan
        gx = t.c + t.a * (np.arange(z.shape[1]) + 0.5)
        gy = t.f + t.e * (np.arange(z.shape[0]) + 0.5)          # descending
        out[key] = RegularGridInterpolator((gx, gy[::-1]), z[::-1].T, bounds_error=False, fill_value=np.nan)
    return out


# ----------------------------------------------------------------------------- layers
def layer_id(name):
    """'D1_THW_SJB2_X47a.txt' -> 'D1';  '0,636ka_THW_...' -> '0,636ka'."""
    m = re.search(r"(?<![A-Za-z0-9])(D\d+)(?![0-9])", name, re.I)
    return m.group(1).upper() if m else name.split("_")[0]


def _is_layer(name):
    n = name.lower()
    return not (n.startswith("surface_") or n.startswith("bed_"))


def collect(Corrected_PST_list):
    """{layer_id: DataFrame(x, y, frac, elev, pst)} plus a DataFrame of all bed-pick points (coverage)."""
    rows, cover = {}, []
    for pst, layers in Corrected_PST_list.items():
        for name, L in layers.items():
            if not (isinstance(L, dict) and "x_array" in L):
                continue
            x, y = np.asarray(L["x_array"], float), np.asarray(L["y_array"], float)
            if not _is_layer(name):
                if name.lower().startswith("bed_"):
                    cover.append(pd.DataFrame({"x": x, "y": y, "pst": pst}))
                continue
            if "frac_depth" not in L:
                raise KeyError(f"{pst}/{name}: no frac_depth — run correct_layers_to_radar_surface first")
            frac = np.asarray(L["frac_depth"], float).copy()
            elev = np.asarray(L["z_array"], float).copy()
            # no radar bed pick nearby: elevation = Bedmap3 surface - radar depth; frac uses Bedmap3 thickness
            if "depth_radar" in L and "surf_bm" in L:
                dep, S = np.asarray(L["depth_radar"], float), np.asarray(L["surf_bm"], float)
                H = np.asarray(L.get("thk_bm", np.full_like(dep, np.nan)), float)
                fill = ~np.isfinite(elev) & np.isfinite(dep) & np.isfinite(S)
                elev[fill] = S[fill] - dep[fill]
                ff = ~np.isfinite(frac) & np.isfinite(dep) & (H > 1)
                frac[ff] = dep[ff] / H[ff]
                if fill.any():
                    print(f"  {layer_id(name)} ({pst}): {fill.sum():,} picks without a radar bed pick within 100 m "
                          "-> Bedmap3 thickness used for their fractional depth")
            d = pd.DataFrame({"x": x, "y": y, "frac": frac, "elev": elev, "pst": pst})
            rows.setdefault(layer_id(name), []).append(d)
    data = {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}
    for k, d in data.items():
        print(f"  {k}: {np.isfinite(d.frac).mean():.0%} of {len(d):,} picks plotted")
    cover = pd.concat(cover, ignore_index=True) if cover else pd.DataFrame(columns=["x", "y", "pst"])
    return data, cover


def _age_label(lid, var=None):
    """Title: date ± uncertainty, then what is plotted."""
    what = {"elev": "Elevation", "frac": "Fractional thickness"}.get(var, "")
    a = AGE_KEY.get(lid)
    if a is None:
        age = lid
    elif lid in AGE_ERR and lid not in NOT_REDATED:
        age = f"{a / 1000:.3f} ± {AGE_ERR[lid] / 1000:.3f} ka"
    else:
        age = f"{a / 1000:.2f} ka"
    return f"{age} · {what}" if what else age


# ----------------------------------------------------------------------------- plotting
def _style():
    dark = BACKGROUND in ("dark", "transparent")
    fg = "white" if dark else "black"
    plt.rcParams.update({
        "font.size": BASE_FONT, "axes.titlesize": BASE_FONT + 6, "axes.labelsize": BASE_FONT + 2,
        "xtick.labelsize": BASE_FONT - 2, "ytick.labelsize": BASE_FONT - 2,
        "axes.linewidth": 2.0, "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 10, "ytick.major.size": 10, "xtick.minor.size": 5, "ytick.minor.size": 5,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5, "xtick.direction": "in", "ytick.direction": "in",
        "text.color": fg, "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg,
    })
    return fg, ("black" if dark else "white"), ("0.45" if dark else "0.78")


def _extent(kind, data, cover):
    tk = np.array(LL2PS.transform(TAKAHE[1], TAKAHE[0])) / 1e3
    if kind == "near":
        cx, cy = tk + np.array(NEAR_CENTER_OFFSET_KM)
        h = NEAR_HALF_KM
        return cx - h * 16 / 9 * 0.8, cx + h * 16 / 9 * 0.8, cy - h, cy + h
    wd = np.array(LL2PS.transform(WAIS_DIVIDE[1], WAIS_DIVIDE[0])) / 1e3
    xs = np.concatenate([d.x.values for d in data.values()] + [cover.x.values]) / 1e3
    ys = np.concatenate([d.y.values for d in data.values()] + [cover.y.values]) / 1e3
    x0, x1 = min(np.nanmin(xs), wd[0], tk[0]) - FULL_PAD_KM, max(np.nanmax(xs), wd[0], tk[0]) + FULL_PAD_KM
    y0, y1 = min(np.nanmin(ys), wd[1], tk[1]) - FULL_PAD_KM, max(np.nanmax(ys), wd[1], tk[1]) + FULL_PAD_KM
    return x0, x1, y0, y1


def _clim(v, var):
    if var in CLIM_FIXED:
        return CLIM_FIXED[var]
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, CLIM_PCT) if v.size else (0, 1)
    return (lo, hi) if hi > lo else (lo - 1, hi + 1)


def plot_layer(lid, d, cover, var, kind, outpath):
    fg, halo_c, track_c = _style()
    halo = [pe.withStroke(linewidth=5, foreground=halo_c)]
    x0, x1, y0, y1 = _extent(kind, {lid: d}, cover) if kind == "near" else _EXT_FULL
    inb = lambda D: (D.x / 1e3 >= x0) & (D.x / 1e3 <= x1) & (D.y / 1e3 >= y0) & (D.y / 1e3 <= y1)
    dd, cc = d[inb(d) & np.isfinite(d[var])], cover[inb(cover)]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    if BACKGROUND == "transparent":
        fig.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor(halo_c)
    ax = fig.add_axes([0.08, 0.14, 0.72, 0.76])
    cax = fig.add_axes([0.84, 0.14, 0.025, 0.76])
    ax.set_facecolor("none" if BACKGROUND == "transparent" else halo_c)

    ax.scatter(cc.x / 1e3, cc.y / 1e3, s=2, c=track_c, lw=0, rasterized=True, zorder=1)
    lo, hi = _clim(dd[var].values, var)
    sp = DOT_SPACING_KM.get(kind, 0) * 1e3
    dp = dd
    if sp > 0:                                         # thin to one pick per grid cell
        key = (dd.x // sp).astype(np.int64) * 1_000_003 + (dd.y // sp).astype(np.int64)
        dp = dd.loc[~key.duplicated()]
    sc = ax.scatter(dp.x / 1e3, dp.y / 1e3, c=dp[var], s=DOT_SIZE.get(kind, 20), cmap=CMAP_FRAC if var == "frac" else CMAP_ELEV,
                    vmin=lo, vmax=hi, rasterized=True, zorder=3)

    tk = np.array(LL2PS.transform(TAKAHE[1], TAKAHE[0])) / 1e3
    wd = np.array(LL2PS.transform(WAIS_DIVIDE[1], WAIS_DIVIDE[0])) / 1e3
    for (px, py), lab, mk, col in ((tk, "Mt. Takahe", "^", "#E4002B"), (wd, "WAIS Divide core", "*", "#FFC20E")):
        if x0 <= px <= x1 and y0 <= py <= y1:
            ax.plot(px, py, mk, ms=26 if mk == "*" else 20, mfc=col, mec=fg, mew=2, zorder=6)
            right = px > x0 + 0.6 * (x1 - x0)                # near the right edge: label to the left
            ax.text(px + (-1 if right else 1) * 0.015 * (x1 - x0), py + 0.015 * (y1 - y0), lab,
                    ha="right" if right else "left", fontsize=BASE_FONT, fontweight="bold",
                    color=fg, path_effects=halo, zorder=7)
    labels = LABEL_LINES if kind == "near" else LABEL_LINES_FULL
    pos = LABEL_POS or {pst: (g.x.iloc[len(g) // 2], g.y.iloc[len(g) // 2])
                        for pst, g in cc.groupby("pst") if len(g) >= 20}
    if labels:
        cx_, cy_ = (x0 + x1) / 2 * 1e3, (y0 + y1) / 2 * 1e3
        for pst, (lx, ly) in pos.items():
            if labels is not True and pst not in labels:
                continue
            lx, ly = np.atleast_1d(lx), np.atleast_1d(ly)
            inside = (lx / 1e3 > x0 + 0.03 * (x1 - x0)) & (lx / 1e3 < x1 - 0.03 * (x1 - x0)) & \
                     (ly / 1e3 > y0 + 0.03 * (y1 - y0)) & (ly / 1e3 < y1 - 0.03 * (y1 - y0))
            if not inside.any():
                continue
            ii = np.where(inside)[0]                         # label out along the line, away from the centre
            r = np.hypot(lx[ii] - cx_, ly[ii] - cy_)
            j = ii[np.argsort(r)[int(0.8 * (len(ii) - 1))]]
            lx, ly = lx[j], ly[j]
            ax.text(lx / 1e3, ly / 1e3, pst, fontsize=BASE_FONT - 6 if kind == "full" else BASE_FONT - 4,
                    color=fg, ha="center", va="center", path_effects=halo, zorder=5)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True)
    ax.set_xlabel("Easting (km, polar stereographic)")
    ax.set_ylabel("Northing (km)")
    where = "flank of Mt. Takahe" if kind == "near" else "Takahe to WAIS Divide"
    what = "Fractional depth" if var == "frac" else "Elevation"
    ax.set_title(_age_label(lid, var), pad=14, fontweight="bold", color=fg)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Fractional depth  (0 = surface, 1 = bed)" if var == "frac" else "Layer elevation (m a.s.l.)",
                 fontsize=BASE_FONT + 2, labelpad=12)
    cb.ax.tick_params(labelsize=BASE_FONT - 2, width=2, length=8, color=fg, labelcolor=fg)
    cb.outline.set_linewidth(2); cb.outline.set_edgecolor(fg)
    if var == "frac":
        cb.ax.invert_yaxis()                        # shallow at the top, like the ice column
    # keep the colorbar the height of the actual map after equal-aspect shrinking
    fig.canvas.draw()
    bb = ax.get_position()
    used = bb.width + 0.02 + 0.025 + 0.09               # map + gap + colorbar + its labels
    shift = (1 - used) / 2 + 0.03 - bb.x0                # centre map+colorbar (0.03 leaves room for y label)
    if shift > 0:
        ax.set_position([bb.x0 + shift, bb.y0, bb.width, bb.height])
        bb = ax.get_position()
    cax.set_position([bb.x1 + 0.02, bb.y0, 0.025, bb.height])

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=DPI, transparent=BACKGROUND == "transparent",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


_EXT_FULL = None
LABEL_POS = {}                        # {line: (x, y)} filled by load_horallpst


# ----------------------------------------------------------------------------- horallpst (x,y,z) exports
def _is_xyz(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return fh.readline().strip().lower().replace(" ", "").startswith("x,y,z")


def _surface_picks(pst_root):
    """Real radar Surface picks from the per-line folders + one label position per line."""
    xs, ys, zs, labels = [], [], [], {}
    for f in sorted(glob.glob(os.path.join(pst_root, "*", "Surface_*"))):
        pst = _pst_token(os.path.basename(f)) or os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f, header=None).iloc[:, :6]
        a, b, z = (pd.to_numeric(d[i], errors="coerce").to_numpy(float) for i in (3, 4, 5))
        x, y = LL2PS.transform(a, b) if COORD == "latlon" else (a, b)
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if ok.sum() == 0:
            continue
        xs.append(x[ok]); ys.append(y[ok]); zs.append(z[ok])
        labels[pst] = (x[ok][::10], y[ok][::10])           # candidate label spots along the line
    if not xs:
        return None, labels
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)), labels


def load_horallpst(root=PICK_ROOT, pst_root=None):
    """
    Reads hor_allpst_*.csv (header x,y,z; EPSG:3031 m; raw radar z) into one pseudo-line 'ALL'
    shaped like produce_layer_df output, and builds a radar 'Surface_' horizon at every bed pick
    (nearest real Surface pick from the per-line folders within SURF_MATCH_M, else SURFACE_RAW_DEFAULT)
    so correct_layers_to_radar_surface can run unchanged.
    """
    from scipy.spatial import cKDTree
    global LABEL_POS
    files = sorted(f for ext in ("*.csv", "*.txt") for f in glob.glob(os.path.join(root, ext)) if _is_xyz(f))
    layers = {}
    for f in files:
        name = os.path.basename(f)
        d = pd.read_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["x", "y", "z"])
        ll = (d.x.abs() <= 90) & (d.y.abs() <= 180) & ((d.x.abs() > 1) | (d.y.abs() > 1))
        if ll.any():                                          # some rows exported as (lat, lon), not metres
            ex, ny = LL2PS.transform(d.loc[ll, "y"].values, d.loc[ll, "x"].values)
            d.loc[ll, "x"], d.loc[ll, "y"] = ex, ny
            print(f"  {name}: {ll.sum():,} picks were in lat/lon; converted to polar stereographic")
        bad = (d.x.abs() < 1) & (d.y.abs() < 1)                  # unpicked traces written at (0, 0)
        if bad.any():
            print(f"  {name}: dropped {bad.sum():,} picks at (0, 0)")
            d = d[~bad]
        lid = "Bed" if re.search(r"(?i)(^|_)bed(_|\.)", name) else layer_id(name)
        layers[f"{lid}_allpst"] = {"x_array": d.x.to_numpy(float), "y_array": d.y.to_numpy(float),
                                   "z_array": d.z.to_numpy(float), "x_position": np.zeros(len(d))}
        print(f"  {name}: {len(d):,} picks -> {lid}")
    if "Bed_allpst" not in layers:
        raise FileNotFoundError("no Bed horizon among the horallpst files")
    # radar surface at every pick location (bed and layers), so depth never depends on a nearby bed pick
    sxa = np.concatenate([L["x_array"] for L in layers.values()])
    sya = np.concatenate([L["y_array"] for L in layers.values()])
    surf, LABEL_POS = _surface_picks(pst_root or PST_ROOT or os.path.dirname(root))
    zs = np.full(sxa.shape, SURFACE_RAW_DEFAULT)
    if surf is not None:
        dist, nn = cKDTree(np.column_stack(surf[:2])).query(np.column_stack([sxa, sya]))
        near = dist <= SURF_MATCH_M
        zs[near] = surf[2][nn[near]]
        print(f"  radar surface: real per-line pick at {near.mean():.0%} of pick locations, "
              f"{SURFACE_RAW_DEFAULT:g} elsewhere")
    layers["Surface_allpst"] = {"x_array": sxa, "y_array": sya, "z_array": zs, "x_position": np.zeros(sxa.size)}
    print(f"loaded {len(files)} horallpst files; line labels from {len(LABEL_POS)} per-line folders")
    return {"ALL": layers}


def make_all(Corrected_PST_list, out_dir=OUT_DIR, layers=None):
    """Writes the 4 sets. `layers` = list like ['D1','D2'] to limit; None = all picked layers."""
    global _EXT_FULL
    data, cover = collect(Corrected_PST_list)
    order = sorted(data, key=lambda k: AGE_KEY.get(k, 1e9))
    if layers:
        order = [k for k in order if k in layers]
    _EXT_FULL = _extent("full", data, cover)
    sets = [("1_fracdepth_Takahe", "frac", "near"), ("2_elevation_Takahe", "elev", "near"),
            ("3_fracdepth_to_WAIS_Divide", "frac", "full"), ("4_elevation_to_WAIS_Divide", "elev", "full")]
    saved = []
    for folder, var, kind in sets:
        for i, lid in enumerate(order, 1):
            p = os.path.join(out_dir, folder, f"{i:02d}_{lid}_{folder.split('_', 1)[1]}.png")
            saved.append(plot_layer(lid, data[lid], cover, var, kind, p))
            print("saved", p)
    return saved


# ----------------------------------------------------------------------------- multi-panel overview
def _grid_title(lid):
    a = AGE_KEY.get(lid)
    if a is None:
        return lid
    if lid in NOT_REDATED or lid not in AGE_ERR:
        return f"{lid}: {a / 1000:.2f} ka (not re-dated)"
    return f"{lid}: {a / 1000:.3f} ± {AGE_ERR[lid] / 1000:.3f} ka"


def plot_grid(data, cover, var, kind, outpath, order):
    """All layers on one 16:9 figure: one map per layer, horizontal colorbar under each."""
    from matplotlib.ticker import MaxNLocator
    fg, halo_c, track_c = _style()
    clear = BACKGROUND == "transparent"
    x0, x1, y0, y1 = _extent("near", data, cover) if kind == "near" else _EXT_FULL
    _in = lambda D: (D.x / 1e3 >= x0) & (D.x / 1e3 <= x1) & (D.y / 1e3 >= y0) & (D.y / 1e3 <= y1)
    empty = [k for k in order if not (_in(data[k]) & np.isfinite(data[k][var])).any()]
    if empty:
        print(f"  [{kind}] no picks in the map window for {', '.join(empty)}; panel skipped")
    order = [k for k in order if k not in empty]
    n = len(order)
    nrow = 2 if n >= 3 else 1
    ncol = int(np.ceil((n + 1) / nrow))                 # +1 slot for the legend
    fs = BASE_FONT - 4
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="none" if clear else halo_c)
    gs = fig.add_gridspec(nrow, ncol, left=0.06, right=0.985, top=0.93, bottom=0.10, wspace=0.18, hspace=0.62)
    tk = np.array(LL2PS.transform(TAKAHE[1], TAKAHE[0])) / 1e3
    wd = np.array(LL2PS.transform(WAIS_DIVIDE[1], WAIS_DIVIDE[0])) / 1e3
    inb = lambda D: (D.x / 1e3 >= x0) & (D.x / 1e3 <= x1) & (D.y / 1e3 >= y0) & (D.y / 1e3 <= y1)
    cc = cover[inb(cover)]
    sp = GRID_DOT_SPACING_KM * 1e3
    for k, lid in enumerate(order):
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("none" if clear else halo_c)
        d = data[lid]
        dd = d[inb(d) & np.isfinite(d[var])]
        if sp > 0 and len(dd):
            key = (dd.x // sp).astype(np.int64) * 1_000_003 + (dd.y // sp).astype(np.int64)
            dp = dd.loc[~key.duplicated()]
        else:
            dp = dd
        ax.scatter(cc.x / 1e3, cc.y / 1e3, s=0.6, c=track_c, lw=0, rasterized=True, zorder=1)
        lo, hi = _clim(dd[var].values, var)
        sc = ax.scatter(dp.x / 1e3, dp.y / 1e3, c=dp[var], s=GRID_DOT_SIZE, rasterized=True, zorder=3,
                        cmap=CMAP_FRAC if var == "frac" else CMAP_ELEV, vmin=lo, vmax=hi)
        for (px, py), mk, col, ms in ((tk, "^", "#E4002B", 13), (wd, "*", "#FFC20E", 17)):
            if x0 <= px <= x1 and y0 <= py <= y1:
                ax.plot(px, py, mk, ms=ms, mfc=col, mec=fg, mew=1.5, zorder=6)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=fg, alpha=0.25, lw=0.8, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=fs - 4, width=1.5, length=6, direction="in", top=True, right=True,
                       color=fg, labelcolor=fg)
        for sp_ in ax.spines.values():
            sp_.set_linewidth(1.5); sp_.set_edgecolor(fg)
        if c != 0:
            ax.set_yticklabels([])
        ax.set_title(_age_label(lid, var), fontsize=fs, fontweight="bold", pad=6, color=fg)
        cax = ax.inset_axes([0.15, -0.20, 0.70, 0.055])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.locator = MaxNLocator(4); cb.update_ticks()
        cb.ax.tick_params(labelsize=fs - 5, width=1.2, length=4, color=fg, labelcolor=fg)
        cb.outline.set_linewidth(1.2); cb.outline.set_edgecolor(fg)
        cb.set_label("Fractional depth" if var == "frac" else "Elevation a.s.l. (m)", fontsize=fs - 4,
                     labelpad=2, color=fg)
    for k in range(n, nrow * ncol):                      # empty slot: legend
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gs[r, c]); ax.axis("off")
        ax.plot([], [], "^", ms=13, mfc="#E4002B", mec=fg, label="Mt. Takahe")
        if kind == "full":
            ax.plot([], [], "*", ms=17, mfc="#FFC20E", mec=fg, label="WAIS Divide core")
        ax.scatter([], [], s=40, c=track_c, label="radar lines, layer not picked")
        leg = ax.legend(loc="center", frameon=False, fontsize=fs - 3)
        for t in leg.get_texts():
            t.set_color(fg)
        ax.text(0.5, 0.12, "x, y: polar stereographic (km)" + ("\nfractional depth: 0 = surface, 1 = bed" if var == "frac" else ""),
                transform=ax.transAxes, ha="center", fontsize=fs - 6, color=fg)
        break
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=DPI, transparent=clear, facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


def make_grids(Corrected_PST_list, out_dir=OUT_DIR, layers=None, data_cover=None):
    """Writes the 4 multi-panel overview figures into out_dir/grids."""
    global _EXT_FULL
    data, cover = data_cover or collect(Corrected_PST_list)
    order = sorted(data, key=lambda k: AGE_KEY.get(k, 1e9))
    if layers:
        order = [k for k in order if k in layers]
    _EXT_FULL = _extent("full", data, cover)
    out = []
    for name, var, kind in (("1_fracdepth_Takahe", "frac", "near"), ("2_elevation_Takahe", "elev", "near"),
                            ("3_fracdepth_to_WAIS_Divide", "frac", "full"), ("4_elevation_to_WAIS_Divide", "elev", "full")):
        p = os.path.join(out_dir, "grids", f"{name}_all_layers.png")
        out.append(plot_grid(data, cover, var, kind, p, order))
        print("saved", p)
    return out


if __name__ == "__main__":
    import sys
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:                                   # some Spyder/console runs don't define __file__
        _here = config.REPO_DIR
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from correct_layers_to_radar_surface import correct_layers_to_radar_surface
    first = sorted(glob.glob(os.path.join(PICK_ROOT, "*.csv")) + glob.glob(os.path.join(PICK_ROOT, "*.txt")))
    PST_list = load_horallpst() if first and _is_xyz(first[0]) else load_picks()
    bm3 = bedmap3_interps(PST_list)
    Corrected = correct_layers_to_radar_surface(PST_list, bm3["surf_interp"], bm3["bed_interp"],
                                                m_per_unit=M_PER_UNIT, firn_corr_m=FIRN_CORR_M,
                                                firn_depth_m=FIRN_DEPTH_M)
    if MAKE_SINGLE:
        make_all(Corrected)
    if MAKE_GRIDS:
        make_grids(Corrected)
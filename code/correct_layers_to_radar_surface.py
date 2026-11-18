# -*- coding: utf-8 -*-
"""
Drop-in replacement for cl.correct_layers_to_absolute_elevation.

Why: the original computes  frac = z_in / thk_bm ;  z_abs = S_bm - frac * thk_bm,
which simplifies to  z_abs = S_bm - z_in.  The raw 'elev' column of the pick files is
measured from a reference ~250 m ABOVE the ice surface (the radar Surface pick itself
reads ~250 m), so every layer ended up ~250 m too low and frac_depth was biased.

This version references every layer to the RADAR surface pick at the nearest trace:
    depth      = z_in(layer) - z_in(surface pick)          [m below ice surface]
    H_radar    = z_in(bed pick) - z_in(surface pick)
    z_array    = S_bm - depth                               [absolute elevation]
    frac_depth = depth / H_radar                            [0 at surface, 1 at bed]
Bedmap3 is used only to anchor the surface elevation (S_bm), not for depths.

Keys added per layer: surf_bm, bed_bm, thk_bm (as before), plus depth_radar, thk_radar,
z_raw (the untouched input). Same array sizes, NaNs preserved.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree


def _nearest_raw(ref_layer, x, y, z_key, max_dist):
    """Raw z of the reference pick (surface or bed) at the nearest trace to each (x, y)."""
    xr = np.asarray(ref_layer["x_array"], float)
    yr = np.asarray(ref_layer["y_array"], float)
    zr = np.asarray(ref_layer[z_key], float)
    ok = np.isfinite(xr) & np.isfinite(yr) & np.isfinite(zr)
    out = np.full(x.shape, np.nan)
    q = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0 or q.sum() == 0:
        return out
    tree = cKDTree(np.column_stack([xr[ok], yr[ok]]))
    dist, nn = tree.query(np.column_stack([x[q], y[q]]), k=1)
    vals = zr[ok][nn]
    vals[dist > max_dist] = np.nan
    out[q] = vals
    return out


def correct_layers_to_radar_surface(PST_list, surf_interp, bed_interp, *,
                                    z_key="z_array", max_dist=100.0, keep_original=True,
                                    m_per_unit=0.8425, firn_corr_m=10.0, firn_depth_m=100.0):
    """
    m_per_unit : metres of ice per Landmark z unit. The LM5 export is TIME domain; the data fit
                 1 unit = 10 ns two-way time -> 168.5 m/us * 0.010 us / 2 = 0.8425 m. Confirm with
                 whoever loaded the radargrams. Set 1.0 to keep raw units.
    firn_corr_m: extra depth added below the firn (faster wave speed in firn makes ice-velocity
                 depths too shallow by roughly this much). Ramps in linearly over firn_depth_m.
                 Set 0 to report ice-velocity depths (closer to ice-equivalent depth).
    """
    PST_src = copy.deepcopy(PST_list) if keep_original else PST_list
    for pst, layers in PST_src.items():
        names = [n for n, v in layers.items() if isinstance(v, dict) and z_key in v]
        surf = max([n for n in names if n.lower().startswith("surface_")], key=len, default=None)
        bed = max([n for n in names if n.lower().startswith("bed_")], key=len, default=None)
        if surf is None or bed is None:
            raise ValueError(f"{pst}: need files starting with 'Surface_' and 'Bed_'")

        # freeze raw values first so surface/bed lookups never see corrected numbers.
        # If the layers were already corrected (cell rerun), reuse the stored raw picks.
        raw = {n: np.asarray(layers[n].get("z_raw", layers[n][z_key]), float).copy() for n in names}
        if all("z_raw" in layers[n] for n in names):
            print(f"[INFO] {pst}: already corrected; recomputing from stored z_raw (safe to rerun)")
        elif any("surf_bm" in layers[n] for n in names):
            raise RuntimeError(
                f"{pst}: layers were already converted by another function (surf_bm present, no z_raw). "
                "Rerun the cell that loads the layers (cl.produce_layer_df) before this one.")
        surf_ref = dict(layers[surf], **{"_raw": raw[surf]})
        bed_ref = dict(layers[bed], **{"_raw": raw[bed]})

        for n in names:
            L = layers[n]
            x = np.asarray(L["x_array"], float)
            y = np.asarray(L["y_array"], float)
            z_in = raw[n]

            S = np.asarray(surf_interp((x, y)), float)
            B = np.asarray(bed_interp((x, y)), float)

            zs = _nearest_raw(surf_ref, x, y, "_raw", max_dist)
            zb = _nearest_raw(bed_ref, x, y, "_raw", max_dist)

            def _to_m(du):
                dm = du * m_per_unit
                return dm + firn_corr_m * np.clip(dm / firn_depth_m, 0.0, 1.0)
            depth = _to_m(z_in - zs)
            H = _to_m(zb - zs)
            ok = np.isfinite(depth) & np.isfinite(H) & (H > 1.0) & np.isfinite(S)

            z_abs = np.full_like(z_in, np.nan)
            frac = np.full_like(z_in, np.nan)
            z_abs[ok] = S[ok] - depth[ok]
            frac[ok] = depth[ok] / H[ok]

            L["z_raw"] = z_in
            L["surf_bm"], L["bed_bm"], L["thk_bm"] = S, B, S - B
            L["depth_radar"], L["thk_radar"] = depth, H
            L["frac_depth"] = frac
            L[z_key] = z_abs

        Hb = layers[bed]["thk_radar"]
        frac_bad = np.mean(~(np.isfinite(Hb) & (Hb > 1.0)))
        if frac_bad > 0.5:
            print(f"[WARN] {pst}: radar thickness invalid for {frac_bad:.0%} of bed points. "
                  "Raw picks may not be depth-like (surface raw should be < bed raw).")
    return PST_src


def thickness_check(PST_corrected, pst):
    """Radar vs Bedmap3 thickness along the bed pick. Ratio ~1 = depths are scaled right."""
    layers = PST_corrected[pst]
    bed = max([n for n in layers if n.lower().startswith("bed_")], key=len)
    L = layers[bed]
    r = L["thk_radar"] / L["thk_bm"]
    r = r[np.isfinite(r)]
    print(f"{pst}: radar H / BM3 H  median {np.median(r):.3f}  "
          f"(5-95%: {np.percentile(r, 5):.3f}-{np.percentile(r, 95):.3f}), n={r.size}")
    return r
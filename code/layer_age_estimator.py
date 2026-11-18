# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:23:21 2026

@author: Jason
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

def estimate_layer_ages(all_layers, Interpolators=None, core_x=None, core_y=None,
                        core_depths=None, core_ages=None, core_err=None,
                        max_dist=2000.0, depth_key="depth_radar", **_ignored):
    """
    Age each layer at its pick nearest the WAIS Divide core, from the layer's radar depth
    below the radar surface pick (depth_radar: metres, firn-corrected by
    correct_layers_to_radar_surface) matched directly to the core depth-age scale.
    No elevations or Bedmap thickness involved. Ages come out in the core's units (ka).
    max_dist: metres; picks farther than this from the core are skipped (and reported).
    """
    import pandas as pd
    cd = np.asarray(core_depths, float); ca = np.asarray(core_ages, float)
    ok = np.isfinite(cd) & np.isfinite(ca)
    cd, ca = cd[ok], ca[ok]
    cd, iu = np.unique(cd, return_index=True); ca = ca[iu]          # sorted, strictly increasing
    ce = np.asarray(core_err, float)[ok][iu] if core_err is not None else None
    age_interp = RGI((cd,), ca, method="linear", bounds_error=False, fill_value=np.nan)

    rows = []
    for old_name, layer in all_layers.items():
        if not (isinstance(layer, dict) and depth_key in layer):
            continue
        x = np.asarray(layer["x_array"], float)
        y = np.asarray(layer["y_array"], float)
        d = np.asarray(layer[depth_key], float)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        if not good.any():
            continue
        dist = np.where(good, np.hypot(x - core_x, y - core_y), np.inf)
        i = int(np.argmin(dist))
        if dist[i] > max_dist:
            print(f"[skip] {old_name}: nearest pick is {dist[i]/1e3:.1f} km from the core")
            continue
        layer_depth = float(d[i])
        age_est = float(age_interp([layer_depth])[0])
        age_err = float(np.interp(layer_depth, cd, ce)) if ce is not None else np.nan
        rows.append((old_name, f"{age_est:.2f}_ka_THW", dist[i], x[i], y[i],
                     layer_depth, age_est, age_err))

    df = pd.DataFrame(rows, columns=["layer_name", "layer age name", "dist_to_core_m",
                                     "x", "y", "layer_depth", "age_est", "age_err"]
                      ).set_index("layer_name")
    
    return df, age_interp
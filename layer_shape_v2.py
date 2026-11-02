# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 21:12:28 2025

@author: Jason
"""

import numpy as np
import matplotlib.pyplot as plt

def shortening(all_layers, x_limits=None, elev_type="true_elevation", plot=True):
    """
    Compute shortening along layers using an *absolute-distance* axis derived from a base layer
    (the layer with the most valid x/y points). This handles radar lines that are not parallel
    to the x-direction.

    Key guarantees:
      - absolute_distance is the SAME LENGTH and INDEXING as each layer's x_array/y_array/z arrays.
      - no resampling / regridding is performed.
      - you can swap between x_array, y_array, absolute_distance (or 3D x/y/z plotting) without
        losing indexing consistency.

    Parameters
    ----------
    all_layers : dict
        Dict of layers. Each layer must have:
          - "x_array", "y_array"
          - either "z_array" (true elevation) or "depth_frac" (fractional depth)
        Arrays may contain NaNs.

    x_limits : tuple/list or None
        (xmin, xmax) in absolute-distance units (same units as x/y).
        If None, user will be prompted in the command window after the plot is shown.

    elev_type : str
        'true_elevation' or 'fractional_depth'

    plot : bool
        If True, scatter plot all layers vs absolute distance.

    Returns
    -------
    shortening_results : dict
        Results keyed by layer name.
        Each result includes:
          - total_length (arc length in distance-z space within window)
          - ref_length   (straight-line distance along the absolute-distance axis within window)
          - shortening_ratio, shortening_percent
          - absdist_window_used
    """

    # ----------------------------
    # 1) Pick base layer = most valid (x,y) points
    # ----------------------------
    def n_valid_xy(layer):
        x = np.asarray(layer["x_array"], dtype=float)
        y = np.asarray(layer["y_array"], dtype=float)
        return np.count_nonzero(np.isfinite(x) & np.isfinite(y))

    layer_items = list(all_layers.items())
    if len(layer_items) == 0:
        print("ERROR: all_layers is empty.")
        return {}

    base_name, base_layer = max(layer_items, key=lambda kv: n_valid_xy(kv[1]))
    x_base = np.asarray(base_layer["x_array"], dtype=float)
    y_base = np.asarray(base_layer["y_array"], dtype=float)

    okb = np.isfinite(x_base) & np.isfinite(y_base)
    if okb.sum() < 2:
        print("ERROR: base layer does not have at least 2 valid (x,y) points.")
        return {}

    # Cumulative distance along base layer (only over valid points; preserve original indexing)
    absdist_base = np.full_like(x_base, np.nan, dtype=float)
    idxb = np.where(okb)[0]
    absdist_base[idxb[0]] = 0.0
    for k in range(1, len(idxb)):
        i0, i1 = idxb[k - 1], idxb[k]
        absdist_base[i1] = absdist_base[i0] + np.hypot(x_base[i1] - x_base[i0], y_base[i1] - y_base[i0])

    # ----------------------------
    # 2) For every layer: map each point -> nearest base point in (x,y) to get absolute distance
    #    (keeps each layer's array size & indexing untouched)
    # ----------------------------
    xb = x_base[okb]
    yb = y_base[okb]
    db = absdist_base[okb]  
    # strictly increasing (in index order over valid points)

    # Pre-pack base coords to speed the nearest-neighbor search (vectorized per point)
    base_xy = np.column_stack([xb, yb])

    def layer_absdist_from_base(layer):
        x = np.asarray(layer["x_array"], dtype=float)
        y = np.asarray(layer["y_array"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)

        out = np.full_like(x, np.nan, dtype=float)
        if ok.sum() == 0:
            return out

        q_xy = np.column_stack([x[ok], y[ok]])  # query points
        # brute-force nearest neighbor (robust, no dependencies)
        # distances^2: (nq, nb)
        d2 = (q_xy[:, None, 0] - base_xy[None, :, 0])**2 + (q_xy[:, None, 1] - base_xy[None, :, 1])**2
        nn = np.argmin(d2, axis=1)
        out[ok] = db[nn]
        return out

    # Build absdist per layer (and store it back into the dict for easy switching later)
    for lname, layer in layer_items:
        layer["absolute_distance"] = layer_absdist_from_base(layer)

    # ----------------------------
    # 3) Plot all layers vs absolute distance (scatter), wide figure
    # ----------------------------
    if plot:
        
        fig, ax = plt.subplots(figsize=(14, 5))  # wider than tall
    
        for lname, layer in layer_items:
            ad = np.asarray(layer["absolute_distance"], dtype=float)
    
            if elev_type == "true_elevation":
                z = np.asarray(layer.get("z_array", np.full_like(ad, np.nan)), dtype=float)
            elif elev_type == "fractional_depth":
                z = np.asarray(layer.get("depth_frac", np.full_like(ad, np.nan)), dtype=float)
            else:
                raise ValueError("Invalid elev_type")
    
            # --- CRITICAL FIX ---
            # Mask invalid values so plot() breaks lines at NaNs
            ad_m = np.ma.masked_invalid(ad)
            z_m  = np.ma.masked_invalid(z)
    
            # Sort by distance so line direction is correct
            order = np.argsort(ad_m)
            ad_m = ad_m[order]
            z_m  = z_m[order]
    
            ax.plot(ad_m, z_m, linewidth=1.25, alpha=0.9, label=lname)
    
        ax.set_xlabel("Absolute distance along radar line")
        ax.set_ylabel(
            "Elevation (m)" if elev_type == "true_elevation" else "Fractional depth"
        )
        ax.set_title("Radar layers vs absolute distance")
        ax.grid(True, alpha=0.25)
    
        # ax.legend(
        #     fontsize=8,
        #     ncol=2,
        #     frameon=False
        # )
    
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # 4) Prompt user for x_limits in absolute distance if not provided
    # ----------------------------
    if x_limits is None:
        # Suggest a reasonable range from the base layer
        dmin = np.nanmin(absdist_base)
        dmax = np.nanmax(absdist_base)
        print(f"\nBase layer: {base_name}")
        print(f"Absolute-distance range (base): [{dmin:.3f}, {dmax:.3f}]")
        raw = input("Enter x-limits as two numbers (xmin xmax) in absolute-distance units: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("ERROR: could not parse two numbers. Example: 1200 5600")
            return {}
        x_limits = (float(parts[0]), float(parts[1]))

    x0, x1 = float(x_limits[0]), float(x_limits[1])
    if x1 < x0:
        x0, x1 = x1, x0  # auto-fix order

    # ----------------------------
    # 5) Compute shortening in window for each layer using (absolute_distance, z)
    # ----------------------------
    shortening_results = {}

    for lname, layer in layer_items:
        ad = np.asarray(layer["absolute_distance"], dtype=float)
        x_pos = np.asarray(layer["x_array"], dtype=float)
        y_pos = np.asarray(layer["y_array"], dtype=float)

        if elev_type == "true_elevation":
            z_arr = np.asarray(layer.get("z_array", np.full_like(ad, np.nan)), dtype=float)
        elif elev_type == "fractional_depth":
            z_arr = np.asarray(layer.get("depth_frac", np.full_like(ad, np.nan)), dtype=float)
        else:
            print("ERROR: please specify a valid elevation type ('true_elevation' or 'fractional_depth')")
            return {}

        # Restrict to window in absolute-distance space
        mask = np.isfinite(ad) & np.isfinite(z_arr) & (ad >= x0) & (ad <= x1)

        if np.count_nonzero(mask) < 2:
            continue

        ad_sub = ad[mask]
        z_sub  = z_arr[mask]

        # Sort by absolute distance to get a consistent traverse direction
        order = np.argsort(ad_sub)
        ad_sub = ad_sub[order]
        z_sub  = z_sub[order]

        # Arc length in (distance, z) space
        dS = np.sqrt(np.diff(ad_sub)**2 + np.diff(z_sub)**2)
        total_length = float(np.nansum(dS))

        # Reference length: straight-line distance along the radar line window (distance axis only)
        # (matches your earlier ref_length = abs(x_limits[1]-x_limits[0]) but now in absolute distance)
        ref_length = float(abs(x1 - x0))

        if total_length <= 0 or ref_length <= 0:
            continue
        if total_length < ref_length:
            # Physically this would imply your curve length < straight distance, usually from noise/windowing.
            # Keep it but you can skip if you prefer.
            pass

        shortening_ratio = (total_length - ref_length) / total_length

        shortening_results[lname] = {
            "total_length": total_length,
            "ref_length": ref_length,
            "shortening_ratio": shortening_ratio,
            "shortening_percent": shortening_ratio * 100.0,
            "absdist_window_used": (x0, x1),
            # Keep these for quick downstream debugging/plotting while preserving indexing
            "absolute_distance": ad,      # full-length array, same indexing
            "x_array": x_pos,
            "y_array": y_pos,
        }

    # Print results
    for k, v in shortening_results.items():
        print(
            f"{k}:\n"
            f" total = {v['total_length']:.3f},\n"
            f" ref   = {v['ref_length']:.3f},\n"
            f" shortening % = {v['shortening_percent']:.6f}\n"
        )
        
    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))  # wider than tall
    
        x0, x1 = x_limits
        if x1 < x0:
            x0, x1 = x1, x0
    
        for lname, layer in layer_items:
            ad_full = np.asarray(layer["absolute_distance"], dtype=float)
    
            # pick z source
            if elev_type == "true_elevation":
                z_full = np.asarray(layer.get("z_array", np.full_like(ad_full, np.nan)), dtype=float)
            elif elev_type == "fractional_depth":
                z_full = np.asarray(layer.get("depth_frac", np.full_like(ad_full, np.nan)), dtype=float)
            else:
                raise ValueError("Invalid elev_type")
    
            # Efficient window mask (per-layer), also drop non-finite so plot doesn't waste time
            m = np.isfinite(ad_full) & np.isfinite(z_full) & (ad_full >= x0) & (ad_full <= x1)
    
            if np.count_nonzero(m) < 2:
                continue
    
            ad = ad_full[m]
            z  = z_full[m]
    
            # Sort by distance so line draws in order
            order = np.argsort(ad)
            ad = ad[order]
            z  = z[order]
    
            # If you still want NaN-break behavior, you can keep masked_invalid,
            # but after finite-filtering above it's not needed.
            ax.plot(ad, z, linewidth=1.25, alpha=0.9, label=lname)
    
        ax.set_xlabel("Absolute distance along radar line")
        ax.set_ylabel("Elevation (m)" if elev_type == "true_elevation" else "Fractional depth")
        ax.set_title(f"Radar layers within x-limits: [{x0:.2f}, {x1:.2f}]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        plt.tight_layout()
        plt.show()
        

    return shortening_results


def max_slope(
    all_layers,
    x_limits=None,
    elev_type="true_elevation",
    axis="absolute_distance",          # "absolute_distance" or "x_array"
    start_index=0,
    plot=True,
    sample_frac=0.60,
    seed=0,
    
    # NEW: slope windowing controls
    slope_method="moving_window",      # "moving_window" or "stride"
    window_dx=None,                    # physical window width in axis units (recommended)
    window_n=None,                     # fallback: window width in number of points (if window_dx is None)
    stride_n=5,                        # only used if slope_method="stride": slope between i and i+stride_n
    min_dx=1e-6,                       # guard for tiny spacing

    # Arrow styling
    arrow_scale=10,
    arrow_width=0.0025,
    arrow_headwidth=4,
    arrow_headlength=6,
    arrow_len_frac=0.50,               # vec_len = (x1-x0)*arrow_len_frac
):
    """
    Re-implementation of max_slope with the same design considerations as `shortening`:

    - Uses a distance-like horizontal axis (default: layer["absolute_distance"]).
    - Prompts the user for x_limits if not provided.
    - Robust to NaNs: never computes slopes across gaps.
    - Preserves array sizes/indexing (no resampling).
    - Sorts by the chosen horizontal coordinate within the window before diffing.
    - Optional fast plotting: randomly samples a fraction of indices *per layer* so x/z
      stay consistent (same indices used across all arrays for that layer).

    Parameters
    ----------
    all_layers : dict
        Layer dict. Each layer has "x_array","y_array", and usually "absolute_distance".
        z-data:
          - elev_type="true_elevation"    -> layer["z_array"]
          - elev_type="fractional_depth"  -> layer["depth_frac"]

    x_limits : (float, float) or None
        Window on the chosen axis (absolute_distance or x_array). If None, prompts user.

    elev_type : str
        "true_elevation" or "fractional_depth"

    axis : str
        "absolute_distance" or "x_array"

    start_index : int
        Skip early entries in all_layers (mimics list(all_layers.items())[2:]).

    plot : bool
        Plot layers again within the selected x_limits.

    sample_frac : float
        Fraction of points to plot (random sample of indices). 0 < sample_frac <= 1.

    seed : int
        RNG seed for reproducible plotting subsamples.

    Returns
    -------
    max_slope_results : dict
        For each layer:
            - max_slope (signed)
            - max_slope_abs
            - x_mid, z_mid (midpoint of the segment where max slope occurs)
            - window_used
            - axis_used
    """
    if axis not in ("absolute_distance", "x_array"):
        raise ValueError("axis must be 'absolute_distance' or 'x_array'")
    if elev_type not in ("true_elevation", "fractional_depth"):
        raise ValueError("elev_type must be 'true_elevation' or 'fractional_depth'")
    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1]")
    if slope_method not in ("moving_window", "stride"):
        raise ValueError("slope_method must be 'moving_window' or 'stride'")
    if slope_method == "moving_window" and window_dx is None and window_n is None:
        # sensible default: 1% of the user window width
        # (keeps behavior stable without needing another prompt)
        pass

    layer_items = list(all_layers.items())[start_index:]

    # ----------------------------
    # Prompt for x_limits if needed
    # ----------------------------
    if x_limits is None:
        # pick a "base" layer for display range: most finite axis points
        def _nfinite_axis(layer):
            a = layer.get(axis, None)
            if a is None:
                return 0
            return np.count_nonzero(np.isfinite(np.asarray(a, dtype=float)))

        base_name, base_layer = max(layer_items, key=lambda kv: _nfinite_axis(kv[1]))
        a0 = np.asarray(base_layer.get(axis, np.array([np.nan])), dtype=float)
        a_min, a_max = np.nanmin(a0), np.nanmax(a0)
        print(f"\nAxis used: {axis}")
        print(f"Example layer for range: {base_name}")
        print(f"{axis} range: [{a_min:.3f}, {a_max:.3f}]")
        raw = input(f"Enter x-limits (xmin xmax) in {axis} units: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("ERROR: could not parse two numbers. Example: 1200 5600")
            return {}
        x_limits = (float(parts[0]), float(parts[1]))

    x0, x1 = map(float, x_limits)
    if x1 < x0:
        x0, x1 = x1, x0
        
    # ----------------------------
    # Helpers for slope computation
    # ----------------------------
    def _max_slope_stride(a, z, stride):
        """Slope between i and i+stride (vectorized)."""
        if a.size <= stride:
            return None
        da = a[stride:] - a[:-stride]
        dz = z[stride:] - z[:-stride]
        good = np.isfinite(da) & np.isfinite(dz) & (np.abs(da) > min_dx)
        if np.count_nonzero(good) == 0:
            return None
        slopes = np.full_like(da, np.nan, dtype=float)
        slopes[good] = dz[good] / da[good]
        k = int(np.nanargmax(np.abs(slopes)))
        i = k
        j = k + stride
        return slopes[k], i, j
    
    def _max_slope_moving_window_dx(a, z, dx_win):
        """
        For each i, pick the smallest j>=i such that a[j]-a[i] >= dx_win,
        then compute slope(i->j). This suppresses tiny-point jitter.
        Uses searchsorted for O(N log N) but cheap in practice.
        """
        n = a.size
        if n < 2:
            return None

        # j indices (first index where a[j] >= a[i] + dx_win)
        j = np.searchsorted(a, a + dx_win, side="left")
        valid = (j < n)

        # compute slopes only where valid
        i_idx = np.where(valid)[0]
        if i_idx.size == 0:
            return None
        j_idx = j[valid]
        da = a[j_idx] - a[i_idx]
        dz = z[j_idx] - z[i_idx]
        
        good = np.isfinite(da) & np.isfinite(dz) & (np.abs(da) > min_dx)
        if np.count_nonzero(good) == 0:
            return None
        
        slopes = np.full_like(da, np.nan, dtype=float)
        slopes[good] = dz[good] / da[good]
        
        kk = int(np.nanargmax(np.abs(slopes)))
        i = int(i_idx[kk])
        jj = int(j_idx[kk])
        return float(slopes[kk]), i, jj
    
    def _max_slope_moving_window_n(a, z, wn):
        """Slope between i and i+wn (vectorized), but wn is in points not distance."""
        if wn <= 0:
            return None
        return _max_slope_stride(a, z, wn)

    # ----------------------------
    # Compute max slope per layer
    # ----------------------------
    max_slope_results = {}

    for fname, layer in layer_items:
        a_full = layer.get(axis, None)
        if a_full is None:
            continue
        a_full = np.asarray(a_full, dtype=float)

        if elev_type == "true_elevation":
            z_full = np.asarray(layer.get("z_array", np.full_like(a_full, np.nan)), dtype=float)
        else:
            z_full = np.asarray(layer.get("depth_frac", np.full_like(a_full, np.nan)), dtype=float)


    # ----------------------------
    # Compute max slope per layer
    # ----------------------------
    max_slope_results = {}

    for fname, layer in layer_items:
        a_full = layer.get(axis, None)
        if a_full is None:
            continue
        a_full = np.asarray(a_full, dtype=float)

        if elev_type == "true_elevation":
            z_full = np.asarray(layer.get("z_array", np.full_like(a_full, np.nan)), dtype=float)
        else:
            z_full = np.asarray(layer.get("depth_frac", np.full_like(a_full, np.nan)), dtype=float)

        # Window + validity. Keep NaNs out so we don't diff across gaps.
        m = np.isfinite(a_full) & np.isfinite(z_full) & (a_full >= x0) & (a_full <= x1)
        if np.count_nonzero(m) < 2:
            continue

        a = a_full[m]
        z = z_full[m]

        # Sort by axis coordinate so diffs are meaningful
        order = np.argsort(a)
        a = a[order]
        z = z[order]

        # # Diffs
        # da = np.diff(a)
        # dz = np.diff(z)

        # # Avoid divide-by-zero / tiny spacing blowups
        # good_seg = np.isfinite(da) & np.isfinite(dz) & (np.abs(da) > 0)
        # if np.count_nonzero(good_seg) == 0:
        #     continue

        # slopes = np.full_like(da, np.nan, dtype=float)
        # slopes[good_seg] = dz[good_seg] / da[good_seg]

        # # Max slope magnitude
        # max_idx = int(np.nanargmax(np.abs(slopes)))
        # max_s = float(slopes[max_idx])
        
        # Choose slope window method
        if slope_method == "stride":
            out = _max_slope_stride(a, z, int(stride_n))
        else:
            if window_dx is not None:
                out = _max_slope_moving_window_dx(a, z, float(window_dx))
            else:
                out = _max_slope_moving_window_n(a, z, int(window_n))
        
        if out is None:
            continue

        max_s, i0, i1 = out

        # Midpoint location of segment (i0 -> i1)
        x_mid = float(0.5 * (a[i0] + a[i1]))
        z_mid = float(0.5 * (z[i0] + z[i1]))

        # Midpoint location of segment
        # x_mid = float(0.5 * (a[max_idx] + a[max_idx + 1]))
        # z_mid = float(0.5 * (z[max_idx] + z[max_idx + 1]))
        
        if slope_method == "moving_window":
            
            max_slope_results[fname] = {
              "max_slope": max_s,
              "max_slope_abs": abs(max_s),
              "x_mid": x_mid,
              "z_mid": z_mid,
              "window_used": (x0, x1),
              "axis_used": axis,
              "slope_window": window_dx,
              "method": slope_method
            }
            
        elif slope_method == "stride":
            
            max_slope_results[fname] = {
            "max_slope": max_s,
            "max_slope_abs": abs(max_s),
            "x_mid": x_mid,
            "z_mid": z_mid,
            "window_used": (x0, x1),
            "axis_used": axis,
            'slope_window': window_n,
            "method": slope_method
            }
            

    # Print results
    for k, v in max_slope_results.items():
        print(
            f"\n{k}:\n"
            f" slope = {v['max_slope']:.6f},\n"
            f" at {axis} = {v['x_mid']:.3f},\n"
            f" z = {v['z_mid']:.6f}",
            f"method used = {v['method']} \n",
            f" window_used = {v['window_used']}"
        )

    # ----------------------------
    # Efficient plot inside limits (sample indices per layer)
    # ----------------------------
    if plot:
        rng = np.random.default_rng(seed)
        fig, ax = plt.subplots(figsize=(14, 5))  # wider than tall

        for fname, layer in layer_items:
            a_full = layer.get(axis, None)
            if a_full is None:
                continue
            a_full = np.asarray(a_full, dtype=float)

            if elev_type == "true_elevation":
                z_full = np.asarray(layer.get("z_array", np.full_like(a_full, np.nan)), dtype=float)
            else:
                z_full = np.asarray(layer.get("depth_frac", np.full_like(a_full, np.nan)), dtype=float)

            # Window mask: keep NaNs OUT for speed. (If you want visual NaN breaks inside
            # the window, remove np.isfinite(z_full) and use masked_invalid below.)
            m = np.isfinite(a_full) & np.isfinite(z_full) & (a_full >= x0) & (a_full <= x1)
            idx = np.where(m)[0]
            if idx.size < 2:
                continue

            # Randomly sample indices (keeps x/z consistent because we sample indices)
            ns = max(2, int(np.ceil(sample_frac * idx.size)))
            if ns < idx.size:
                idx_s = rng.choice(idx, size=ns, replace=False)
            else:
                idx_s = idx

            a = a_full[idx_s]
            z = z_full[idx_s]

            # Sort so plot draws a coherent line
            order = np.argsort(a)
            a = a[order]
            z = z[order]

            # Mask invalid (extra safety; shouldn't be needed given m above)
            a_m = np.ma.masked_invalid(a)
            z_m = np.ma.masked_invalid(z)

            ax.plot(a_m, z_m, linewidth=1.1, alpha=0.9, label=fname)
            
            if fname in max_slope_results:
                x_pt = max_slope_results[fname]["x_mid"]
                z_pt = max_slope_results[fname]["z_mid"]
                slope = max_slope_results[fname]["max_slope"]

                vec_len = (x1 - x0) * float(arrow_len_frac)
                dx_vec = vec_len / np.sqrt(1.0 + slope**2)
                dz_vec = slope * dx_vec

                ax.quiver(
                    x_pt, z_pt, dx_vec, dz_vec,
                    angles="xy", scale_units="xy", scale=arrow_scale,
                    color="white", width=arrow_width,
                    headwidth=arrow_headwidth, headlength=arrow_headlength
                )
            else:
                print("ERROR: Layer name not carried over into slope_results")

        ax.set_xlabel(axis)
        ax.set_ylabel("Elevation (m)" if elev_type == "true_elevation" else "Fractional depth")
        title_win = f"{axis} in [{x0:.3f}, {x1:.3f}]"
        if slope_method == "moving_window":
            title_s = f"moving_window dx={window_dx:.3f}" if window_dx is not None else f"moving_window n={window_n}"
        else:
            title_s = f"stride n={stride_n}"
        ax.set_title(f"Max slope ({title_s}) | {title_win} | plotted {int(sample_frac*100)}% points")
        ax.grid(True, alpha=0.25)

        # ax.legend(fontsize=8, ncol=2, frameon=False)
        plt.tight_layout()
        plt.show()

    return max_slope_results



def fold_extrema(
    all_layers,
    x_limits=None,
    elev_type="true_elevation",
    axis="absolute_distance",          # "absolute_distance" or "x_array"
    start_index=0,
    plot=True,
    sample_frac=0.70,
    seed=0,

    # Extrema finding controls
    extrema_mode="global",             # "global" or "robust"
    edge_trim_frac=0.00,              # trim this fraction off each end before extrema (0..0.2 typical)
    min_points=5,

    # Optional fold annotation
    annotate_fold=False,
    show_height=False,                  # dotted vertical line for amplitude
    show_half_wavelength=False,          # dotted horizontal line for (wavelength/2)
    invert_yaxis=False,                # handy for fractional depth plots
    dot_size=28,
    line_alpha=0.9,
):
    """
    Finds max/min z within a user-defined window on each layer (for fold amplitude & wavelength proxies),
    using the same general framework as your shortening/max_slope tools:

    - Uses axis (default: absolute_distance) so curved radar lines work.
    - Prompts for x_limits if not provided.
    - Preserves indexing (no resampling). Works directly on original arrays.
    - Optional fast plotting: randomly samples indices per layer to plot.
    - Computes per-layer:
        * z_max, z_min
        * x_at_zmax, x_at_zmin  (axis coordinate values; typically abs distance)
        * amplitude = z_max - z_min
        * half_wavelength = 0.5 * |x_at_zmax - x_at_zmin|
        * wavelength_proxy = |x_at_zmax - x_at_zmin|
      (This “wavelength” is a proxy based on peak-to-trough spacing inside the window.)

    Final plot:
    - Red dots at maxima, blue dots at minima.
    - Optional dotted lines: vertical (height) and horizontal (half-wavelength).

    Parameters
    ----------
    all_layers : dict
        Must contain per-layer arrays:
            - axis array: layer["absolute_distance"] or layer["x_array"]
            - z array: layer["z_array"] or layer["depth_frac"]
    x_limits : (float, float) or None
        If None, prompts user for xmin xmax.
    elev_type : "true_elevation" | "fractional_depth"
    axis : "absolute_distance" | "x_array"
    sample_frac : float
        Fraction of in-window points to plot per layer (for speed).
    extrema_mode : "global" | "robust"
        "global": raw max/min within window.
        "robust": uses percentiles to avoid one-point spikes (still returns actual nearest points).
    edge_trim_frac : float
        Trims off edge_trim_frac of the *windowed points* at both ends before extrema.
        Helps avoid edge artifacts.
    annotate_fold : bool
        If True, draw dotted lines showing amplitude and half-wavelength.
    """

    if axis not in ("absolute_distance", "x_array"):
        raise ValueError("axis must be 'absolute_distance' or 'x_array'")
    if elev_type not in ("true_elevation", "fractional_depth"):
        raise ValueError("elev_type must be 'true_elevation' or 'fractional_depth'")
    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1]")
    if extrema_mode not in ("global", "robust"):
        raise ValueError("extrema_mode must be 'global' or 'robust'")
    if not (0.0 <= edge_trim_frac < 0.5):
        raise ValueError("edge_trim_frac must be in [0, 0.5)")

    layer_items = list(all_layers.items())[start_index:]
    
    print(all_layers.items())

    # ----------------------------
    # Prompt for x_limits if needed
    # ----------------------------
    if x_limits is None:
        def _nfinite_axis(layer):
            a = layer.get(axis, None)
            if a is None:
                return 0
            return np.count_nonzero(np.isfinite(np.asarray(a, dtype=float)))

        base_name, base_layer = max(layer_items, key=lambda kv: _nfinite_axis(kv[1]))
        a0 = np.asarray(base_layer.get(axis, np.array([np.nan])), dtype=float)
        a_min, a_max = np.nanmin(a0), np.nanmax(a0)

        print(f"\nAxis used: {axis}")
        print(f"Example layer for range: {base_name}")
        print(f"{axis} range: [{a_min:.3f}, {a_max:.3f}]")
        raw = input(f"Enter x-limits (xmin xmax) in {axis} units: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("ERROR: could not parse two numbers. Example: 50000 100000")
            return {}

        x_limits = (float(parts[0]), float(parts[1]))

    x0, x1 = map(float, x_limits)
    if x1 < x0:
        x0, x1 = x1, x0

    rng = np.random.default_rng(seed)

    # ----------------------------
    # Compute extrema per layer
    # ----------------------------
    results = {}

    for fname, layer in layer_items:
        
        print(fname)
        
        a_full = layer.get(axis, None)
        if a_full is None:
            continue
        a_full = np.asarray(a_full, dtype=float)

        if elev_type == "true_elevation":
            z_full = np.asarray(layer.get("z_array", np.full_like(a_full, np.nan)), dtype=float)
        else:
            z_full = np.asarray(layer.get("depth_frac", np.full_like(a_full, np.nan)), dtype=float)

        # Window + validity
        m = np.isfinite(a_full) & np.isfinite(z_full) & (a_full >= x0) & (a_full <= x1)
        idx = np.where(m)[0]
        if idx.size < min_points:
            continue

        a = a_full[idx]
        z = z_full[idx]

        # Sort by axis coordinate (important for trimming, plotting, etc.)
        order = np.argsort(a)
        a = a[order]
        z = z[order]
        idx_sorted = idx[order]  # indices into original arrays (preserves mapping)

        # Optional edge trim
        if edge_trim_frac > 0.0 and a.size >= 2 * min_points:
            k = int(np.floor(edge_trim_frac * a.size))
            if (a.size - 2 * k) >= min_points:
                a = a[k:-k]
                z = z[k:-k]
                idx_sorted = idx_sorted[k:-k]

        if a.size < min_points:
            continue

        # Extrema selection
        if extrema_mode == "global":
            i_max = int(np.nanargmax(z))
            i_min = int(np.nanargmin(z))
        else:
            # Robust mode: target upper/lower percentiles then pick nearest actual points
            z_hi = np.nanpercentile(z, 97.5)
            z_lo = np.nanpercentile(z, 2.5)
            i_max = int(np.nanargmin(np.abs(z - z_hi)))
            i_min = int(np.nanargmin(np.abs(z - z_lo)))

        x_at_max = float(a[i_max])
        z_max = float(z[i_max])
        x_at_min = float(a[i_min])
        z_min = float(z[i_min])

        amp = z_max - z_min
        dx_peak_trough = abs(x_at_max - x_at_min)
        half_wavelength = 0.5 * dx_peak_trough

        results[fname] = {
            "z_max": z_max,
            "x_at_zmax": x_at_max,
            "z_min": z_min,
            "x_at_zmin": x_at_min,
            "amplitude": amp,
            "dx_peak_trough": dx_peak_trough,
            "half_wavelength": half_wavelength,
            "window_used": (x0, x1),
            "axis_used": axis,
            "elev_type": elev_type,
            "i_max_original": int(idx_sorted[i_max]),
            "i_min_original": int(idx_sorted[i_min]),
        }

    # Print summary
    for k, v in results.items():
        print(
            f"\n{k}\n"
            f"  max z = {v['z_max']:.4f} at {axis} = {v['x_at_zmax']:.2f}\n"
            f"  min z = {v['z_min']:.4f} at {axis} = {v['x_at_zmin']:.2f}\n"
            f"  amplitude (max-min) = {v['amplitude']:.4f}\n"
            f"  dx(peak-trough) = {v['dx_peak_trough']:.2f} | half = {v['half_wavelength']:.2f}"
        )

    # ----------------------------
    # Plot (sampled) + extrema markers + optional dotted guides
    # ----------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))  # wide

        for fname, layer in layer_items:
            a_full = layer.get(axis, None)
            if a_full is None:
                continue
            a_full = np.asarray(a_full, dtype=float)

            if elev_type == "true_elevation":
                z_full = np.asarray(layer.get("z_array", np.full_like(a_full, np.nan)), dtype=float)
            else:
                z_full = np.asarray(layer.get("depth_frac", np.full_like(a_full, np.nan)), dtype=float)

            m = np.isfinite(a_full) & np.isfinite(z_full) & (a_full >= x0) & (a_full <= x1)
            idx = np.where(m)[0]
            if idx.size < 2:
                continue

            # Sample indices for plotting speed (keeps consistency)
            ns = max(2, int(np.ceil(sample_frac * idx.size)))
            idx_s = rng.choice(idx, size=ns, replace=False) if ns < idx.size else idx

            a = a_full[idx_s]
            z = z_full[idx_s]
            order = np.argsort(a)
            a = a[order]
            z = z[order]

            # Fast line plot; masked_invalid prevents stray connections if any invalid slip through
            ax.plot(np.ma.masked_invalid(a), np.ma.masked_invalid(z),
                    linewidth=1.0, alpha=line_alpha)

            # Extrema markers + guides
            if fname in results:
                r = results[fname]

                # Max in red, min in blue
                ax.scatter(r["x_at_zmax"], r["z_max"], c="red", s=dot_size, zorder=5)
                ax.scatter(r["x_at_zmin"], r["z_min"], c="blue", s=dot_size, zorder=5)

                if annotate_fold:
                    # Vertical dotted line showing amplitude (height)
                    if show_height:
                        ax.plot(
                            [r["x_at_zmax"], r["x_at_zmax"]],
                            [r["z_min"], r["z_max"]],
                            linestyle=":", linewidth=1.2, alpha=0.9, color="white"
                        )

                    # Horizontal dotted line showing half-wavelength from midpoint to peak (or trough)
                    if show_half_wavelength:
                        x_mid = 0.5 * (r["x_at_zmax"] + r["x_at_zmin"])
                        # draw half-wavelength on the mid-elevation between peak and trough
                        z_mid = 0.5 * (r["z_max"] + r["z_min"])
                        ax.plot(
                            [x_mid, r["x_at_zmax"]],
                            [z_mid, z_mid],
                            linestyle=":", linewidth=1.2, alpha=0.9, color="white"
                        )
                        ax.plot(
                            [x_mid, r["x_at_zmin"]],
                            [z_mid, z_mid],
                            linestyle=":", linewidth=1.2, alpha=0.9, color="white"
                        )

        ax.set_xlim(x0, x1)
        ax.set_xlabel(axis)
        ax.set_ylabel("Elevation (m)" if elev_type == "true_elevation" else "Fractional depth")
        ax.set_title(
            f"Fold extrema within [{x0:.2f}, {x1:.2f}] ({axis}) | plotted {int(sample_frac*100)}% points"
        )
        ax.grid(True, alpha=0.25)
        if invert_yaxis:
            ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

    return results


# =============================================================================
# Layer spacing (isopach) along the line
# =============================================================================

# Landmark export lettering -> ages (years). Edit here if the key changes.
AGE_KEY = {"D1": 640.0, "D2": 1210.0, "D3": 2100.0, "D4": 2240.0,
           "D5": 2760.0, "D6": 4320.0, "D7": 1800.0}


def _age_from_key(name, age_key):
    """Find a 'D<n>' token in the name (e.g. 'D3_THW_SJB2_X47a.txt') and look it up."""
    import re
    if not age_key:
        return None
    m = re.search(r"(?<![A-Za-z0-9])(D\d+)(?![0-9])", name, re.I)
    if m and m.group(1).upper() in {k.upper(): v for k, v in age_key.items()}:
        return float({k.upper(): v for k, v in age_key.items()}[m.group(1).upper()])
    return None


def _layer_age_yr(name, age_key=None):
    """Age in years from the Landmark letter key (D1..D7) or from names like '0,64ka_...'."""
    import re
    a = _age_from_key(name, AGE_KEY if age_key is None else age_key)
    if a is not None:
        return a
    m = re.match(r"\s*(\d+)[,.](\d+)\s*(ka)?", name, re.I)
    return float(f"{m.group(1)}.{m.group(2)}") * 1000 if m else None


def layer_spacing(
    all_layers,
    positions,
    axis=None,                # None = use "abs_distance" (master script) or "absolute_distance" (shortening())
    z_key="z_array",
    layers=None,              # optional list of layer names, top -> bottom; default = dated layers by age
    include_surface=True,     # add a surface -> youngest layer interval
    include_bed=False,        # add an oldest layer -> bed interval
    include_surface_bed=False,  # old switch: same as include_surface=True, include_bed=True
    max_gap=200.0,            # max distance (axis units) to the nearest pick before a layer counts as absent there
    normalize=False,          # also return thickness / median thickness of each interval (relative thickening)
    plot=True,
    age_key=None,             # {"D1": 640, ...}; None = AGE_KEY at the top of this section
    fig_width=20,             # inches; height scales with the number of panels
    savepath=None,            # e.g. "X47a_spacing.png" -> saved with a transparent background
    title=None,               # e.g. the PST name; shown at the start of the figure title
    background=None,          # None/"transparent" = no background (follows your theme), "light" = white, "dark" = black
    outdir=None,              # folder to save into; file name = <title>_spacing_<background>.png
):
    """
    Vertical distance between consecutive layers at chosen positions along the line.

    positions : 1D array of axis values (e.g. np.linspace(170e3, 200e3, 301)) or a list of points.
    Returns dict:
        "positions": positions,
        "intervals": {"0.64-1.80 ka": thickness_array, ...}   (m, NaN where a layer is absent)
        "relative" : same keys, thickness / median (only if normalize=True)
        "layer_z"  : {layer_name: z interpolated at positions}
    Layers are ordered young -> old using the ages in their names (or the order you pass in).
    Interpolation is linear along the axis; nothing is resampled in the original arrays.
    """
    positions = np.atleast_1d(np.asarray(positions, dtype=float))
    if axis is None:
        dicts = [v for v in all_layers.values() if isinstance(v, dict)]
        axis = next((k for k in ("abs_distance", "absolute_distance") if any(k in d for d in dicts)), None)
        if axis is None:
            raise ValueError("No 'abs_distance' or 'absolute_distance' found; pass axis=...")

    # real layers only (skip PST-level arrays like bmelt, W, X ...)
    items = {k: v for k, v in all_layers.items() if isinstance(v, dict) and axis in v and z_key in v}
    if layers is None:
        dated = [(k, _layer_age_yr(k, age_key)) for k in items]
        names = [k for k, a in sorted(((k, a) for k, a in dated if a is not None), key=lambda t: t[1])]
        surf = [k for k in items if k.lower().startswith("surface")]
        bed = [k for k in items if k.lower().startswith("bed")]
        if include_surface or include_surface_bed:
            names = surf[:1] + names
        if include_bed or include_surface_bed:
            names = names + bed[:1]
    else:
        names = list(layers)
    if len(names) < 2:
        dicts = {k: sorted(v.keys()) for k, v in all_layers.items() if isinstance(v, dict)}
        sample = next(iter(dicts.items()), ("(none)", []))
        raise ValueError(
            f"Need at least two dated layers; found {names}.\n"
            f"  axis used: '{axis}', z_key: '{z_key}'\n"
            f"  entries with both keys: {list(items.keys())}\n"
            f"  ages parsed from names: {[(k, _layer_age_yr(k, age_key)) for k in items]}\n"
            f"  keys in first layer ('{sample[0]}'): {sample[1]}\n"
            f"  Fix: pass axis=... / z_key=... to match those keys, or layers=[...] in top-to-bottom order.")

    def _z_at(name):
        L = items[name]
        a = np.asarray(L[axis], float); z = np.asarray(L[z_key], float)
        ok = np.isfinite(a) & np.isfinite(z)
        if ok.sum() < 2:
            return np.full(positions.shape, np.nan)
        a, z = a[ok], z[ok]
        o = np.argsort(a); a, z = a[o], z[o]
        out = np.interp(positions, a, z, left=np.nan, right=np.nan)
        # blank positions where the nearest pick is too far away (gaps in the layer)
        j = np.clip(np.searchsorted(a, positions), 1, a.size - 1)
        near = np.minimum(np.abs(a[j] - positions), np.abs(a[j - 1] - positions))
        out[near > max_gap] = np.nan
        return out

    layer_z = {n: _z_at(n) for n in names}

    def _label(n):
        a = _layer_age_yr(n, age_key)
        if a is not None:
            return f"{a/1000:.2f}"
        return "surface" if n.lower().startswith("surface") else "bed" if n.lower().startswith("bed") else n

    intervals, relative = {}, {}
    for upper, lower in zip(names[:-1], names[1:]):
        key = f"{_label(upper)}-{_label(lower)}" + (" ka" if _layer_age_yr(upper, age_key) or _layer_age_yr(lower, age_key) else "")
        th = layer_z[upper] - layer_z[lower]
        intervals[key] = th
        if normalize:
            med = np.nanmedian(th)
            relative[key] = th / med if np.isfinite(med) and med != 0 else th * np.nan

    for k, th in intervals.items():
        v = th[np.isfinite(th)]
        if v.size:
            print(f"{k}: n={v.size}, min {v.min():.0f} m, median {np.median(v):.0f} m, max {v.max():.0f} m")
        else:
            print(f"{k}: no overlap of both layers at these positions")

    # ice thickness from the Surface_ / Bed_ picks (same interpolation and gap rule as the layers)
    surf_name = next((k for k in items if k.lower().startswith("surface")), None)
    bed_name = next((k for k in items if k.lower().startswith("bed")), None)
    surf_z = _z_at(surf_name) if surf_name else None
    bed_z = _z_at(bed_name) if bed_name else None
    ice_thk = (surf_z - bed_z) if (surf_z is not None and bed_z is not None) else None

    if plot:
        import contextlib
        transparent = background not in ("light", "dark")
        style = {"light": "default", "dark": "dark_background"}.get(background)
        ctx = plt.style.context(style) if style else contextlib.nullcontext()
        with ctx:
            from matplotlib.ticker import AutoMinorLocator
            fg = plt.rcParams.get("text.color", "k")         # follows your theme (white on dark, black on light)
            colors = {k: plt.cm.tab10(i % 10) for i, k in enumerate(intervals)}
            few = positions.size <= 20                      # discrete points -> draw markers
            sty = dict(lw=2.0, marker="o" if few else None, ms=6)
            lo, hi = np.nanmin(positions), np.nanmax(positions)
            pad = 0.02 * (hi - lo) if hi > lo else 1000.0

            def _raw(name):
                """Original picks inside the plotted window (not the interpolated values)."""
                L = items[name]
                a = np.asarray(L[axis], float); z = np.asarray(L[z_key], float)
                m = np.isfinite(a) & np.isfinite(z) & (a >= lo - pad) & (a <= hi + pad)
                o = np.argsort(a[m])
                return a[m][o], z[m][o]

            panels = ["layers"] + (["ice"] if ice_thk is not None else []) + ["thk"] + (["rel"] if normalize else [])
            heights = {"layers": 2.2, "ice": 1.0, "thk": 1.6, "rel": 1.6}
            fig, axs = plt.subplots(len(panels), 1, figsize=(fig_width, 3.4 * sum(heights[p] for p in panels)),
                                    sharex=True, gridspec_kw={"height_ratios": [heights[p] for p in panels]})
            axs = np.atleast_1d(axs)
            A = dict(zip(panels, axs))

            # 1) the layers themselves, with each interval shaded in the colour used below
            ax = A["layers"]
            for (k, _), upper, lower in zip(intervals.items(), names[:-1], names[1:]):
                ax.fill_between(positions, layer_z[upper], layer_z[lower], color=colors[k], alpha=0.30, lw=0)
            for n in names:
                a, z = _raw(n)
                ax.plot(a, z, color=fg, lw=1.2)
                if a.size:
                    ax.annotate(_label(n) + (" ka" if _layer_age_yr(n, age_key) else ""), (a[-1], z[-1]),
                                xytext=(6, 0), textcoords="offset points", fontsize=11, va="center")
            for n, c in ((surf_name, "steelblue"), (bed_name, "saddlebrown")):
                if n and n not in names:
                    a, z = _raw(n)
                    ax.plot(a, z, color=c, lw=2.0, label="surface" if c == "steelblue" else "bed")
            if few:
                for p in positions:
                    ax.axvline(p, color="0.5", lw=0.8, ls=":")
            ax.set_ylabel("Elevation (m)" if z_key == "z_array" else z_key)
            ax.set_title((f"{title}: " if title else "") + "Layers along the line (shading = intervals below)")
            if any(n and n not in names for n in (surf_name, bed_name)):
                ax.legend(frameon=False, loc="lower left")

            # 2) ice thickness
            if "ice" in A:
                ax = A["ice"]
                ax.plot(positions, ice_thk, color=fg, **sty)
                ax.set_ylabel("Ice thickness (m)")

            # 3) interval thickness
            ax = A["thk"]
            for k, th in intervals.items():
                ax.plot(positions, th, color=colors[k], label=k, **sty)
            ax.set_ylabel("Interval thickness (m)")
            ax.legend(frameon=False, ncol=3, loc="best")

            # 4) relative thickness
            if normalize:
                ax = A["rel"]
                for k, r in relative.items():
                    ax.plot(positions, r, color=colors[k], label=k, **sty)
                ax.axhline(1, color=fg, lw=1.0, ls="--")
                ax.set_ylabel("Thickness / median")

            # big-screen formatting: no background, major + minor grid, ticks inside on all four sides
            if transparent:
                fig.patch.set_alpha(0.0)
            for ax in axs:
                if transparent:
                    ax.set_facecolor("none")
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.grid(True, which="major", alpha=0.35, lw=0.9)
                ax.grid(True, which="minor", alpha=0.12, lw=0.6, ls=":")
                ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
                ax.tick_params(which="major", length=8, width=1.1)
                ax.tick_params(which="minor", length=4, width=0.8)
                ax.ticklabel_format(axis="both", style="plain", useOffset=False)
                ax.yaxis.label.set_size(14)
                for t in ax.get_legend().get_texts() if ax.get_legend() else []:
                    t.set_fontsize(12)
            axs[0].title.set_size(16)
            axs[-1].set_xlabel(axis, fontsize=14)
            axs[-1].set_xlim(lo - pad, hi + pad)
            fig.align_ylabels(axs)
            plt.tight_layout()
            if outdir and not savepath:
                import os
                os.makedirs(outdir, exist_ok=True)
                tag = background if background in ("light", "dark") else "transparent"
                savepath = os.path.join(outdir, f"{title or 'layer_spacing'}_spacing_{tag}.png")
            if savepath:
                fig.savefig(savepath, dpi=200, transparent=transparent, bbox_inches="tight",
                            facecolor=None if transparent else fig.get_facecolor())
                print(f"saved {savepath}")
            plt.show()

    out = {"positions": positions, "intervals": intervals, "layer_z": layer_z, "ice_thickness": ice_thk}
    if normalize:
        out["relative"] = relative
    return out
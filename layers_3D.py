# -*- coding: utf-8 -*-
"""
layers_3d.py — 3D view of Bedmap3 bed + ice surface with radar isochrones.

Uses plot_Surface_3D (flowband functions) for the bed, then adds:
  - a semi-transparent ice surface
  - every layer in Corrected_PST_list as a 3D line (absolute elevation, z_array),
    coloured by age, with the radar Surface/Bed picks drawn in black/grey
  - optional point markers (e.g. Mt. Takahe, monocline dip picks)

Grid is built by evaluating your Bedmap3 interpolators over the PSTs' extent, so it
works with whatever load_bedmap3 returns (only bm3["bed_interp"] / ["surf_interp"] are used).
"""
import re
import numpy as np
import plotly.graph_objects as go

AGE_COLORS = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]



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


def _age_years(name):
    a = _age_from_key(name, AGE_KEY)
    if a is not None:
        return a
    m = re.match(r"\s*(\d+)[,.](\d+)\s*(ka)?", name, re.I)
    return float(f"{m.group(1)}.{m.group(2)}") * 1000 if m else None


def _layer_items(pst_dict):
    """Yield (name, layer_dict) for real layers, skipping PST-level arrays (bmelt, W, X, ...)."""
    for name, val in pst_dict.items():
        if isinstance(val, dict) and "x_array" in val and "z_array" in val:
            yield name, val


def plot_layers_3d(Corrected_PST_list, bm3, plot_surface_3d, *,
                   psts=None, limits_by_pst=None, res=1000.0, pad=15e3,
                   vert_exag=25.0, step=5, show_ice_surface=True,
                   markers=None, flowlines=None, label_ages=True,
                   vel_interp=None, speed_on=None, speed_contours=None, speed_max=None, speed_opacity=0.6,
                   surface_contours=100.0, surface_contour_opacity=0.45, surface_contour_color="#2b2b2b",
                   title="Bed, ice surface and dated isochrones"):
    """
    Corrected_PST_list : Data.Corrected_PST_list / PST_layers (after the correction)
    bm3                : output of load_bedmap3 (needs bm3["bed_interp"], bm3["surf_interp"])
    plot_surface_3d    : your plot_Surface_3D function (passed in so this file has no hard import)
    psts               : list of PST names to draw (default: all)
    limits_by_pst      : {pst: (xmin, xmax)} in abs_distance metres, e.g. from ArgumentsByPST
    res, pad           : grid spacing and padding around the PSTs (m)
    vert_exag          : vertical exaggeration (25 = 25x)
    step               : plot every `step`-th radar point (17 m spacing -> 5 gives ~85 m)
    markers            : {label: (x, y)} in EPSG:3031 metres; drawn on the ice surface
    flowlines          : {label: (x_array, y_array)} e.g. from strain_tools.forward_flowline;
                         draped on the ice surface (optional third array = travel time, shown on hover)
    label_ages         : write the age at the end of each isochrone line
    vel_interp         : surface-speed interpolator called as f((x, y)) in m/yr
                         (e.g. Interpolators.Vel). Needed for the two options below.
    speed_on           : None, "bed" or "surface" -> colour that surface by ice speed
    speed_contours     : list of speeds (m/yr) to draw as contour lines on the ice surface,
                         e.g. [5, 10, 25, 50, 100]
    speed_max          : top of the speed colour scale (default: 98th percentile)
    speed_opacity      : opacity of the speed-coloured ice surface (lower = see layers through it)
    surface_contours   : elevation-contour interval (m) drawn on the ice surface; None = off
    surface_contour_opacity : contour-line opacity (surface is 0.25, or speed_opacity when speed-coloured)
    surface_contour_color   : contour-line colour
    """
    psts = psts or list(Corrected_PST_list.keys())

    # ---- collect layer points (respecting optional limits)
    lines, xs, ys = [], [], []
    for pst in psts:
        for name, L in _layer_items(Corrected_PST_list[pst]):
            x = np.asarray(L["x_array"], float)
            y = np.asarray(L["y_array"], float)
            z = np.asarray(L["z_array"], float)
            keep = np.isfinite(x) & np.isfinite(y)
            if limits_by_pst and pst in limits_by_pst and "abs_distance" in L:
                a = np.asarray(L["abs_distance"], float)
                lo, hi = limits_by_pst[pst]
                keep &= (a >= lo) & (a <= hi)
            if keep.sum() < 2:
                continue
            idx = np.where(keep)[0][::step]
            zz = z[idx].copy()
            # break the line where points were dropped so Plotly doesn't bridge gaps
            gaps = np.where(np.diff(idx) > step * 3)[0]
            xl, yl, zl = list(x[idx]), list(y[idx]), list(zz)
            for g in gaps[::-1]:
                for arr in (xl, yl, zl):
                    arr.insert(g + 1, None)
            lines.append((pst, name, xl, yl, zl))
            xs.append(x[idx]); ys.append(y[idx])
    if not lines:
        raise ValueError("No layer points found for the requested PSTs / limits.")

    # ---- Bedmap3 grid over the layers' extent
    xa, ya = np.concatenate(xs), np.concatenate(ys)
    gx = np.arange(np.nanmin(xa) - pad, np.nanmax(xa) + pad + res, res)
    gy = np.arange(np.nanmin(ya) - pad, np.nanmax(ya) + pad + res, res)
    GX, GY = np.meshgrid(gx, gy)
    bed = np.asarray(bm3["bed_interp"]((GX, GY)), float)
    srf = np.asarray(bm3["surf_interp"]((GX, GY)), float)

    # aspect ratio from a real vertical exaggeration
    xspan, yspan = np.ptp(gx), np.ptp(gy)
    zspan = np.nanmax(srf) - np.nanmin(bed)
    xr = 1.3
    exag = xr * (zspan / xspan) * vert_exag

    # ---- bed via your function (it returns the figure without showing it)
    fig = plot_surface_3d(bed, GX, GY, gx, gy, "Bed", bed, exag=exag, contours=False, plot2D=False)
    fig.update_traces(selector=dict(type="surface"), name="Bedmap3 bed", showlegend=True,
                      colorbar=dict(title="Bed (m)", x=1.0, len=0.6))

    # ---- ice speed on the same grid (optional)
    spd = None
    if vel_interp is not None:
        spd = np.asarray(vel_interp((GX, GY)), float)
        smax = speed_max or float(np.nanpercentile(spd, 98))
        speed_cbar = dict(title="Speed (m/yr)", x=1.0, len=0.6)
    if spd is not None and speed_on == "bed":
        fig.update_traces(selector=dict(type="surface"), surfacecolor=spd, colorscale="Viridis",
                          cmin=0, cmax=smax, colorbar=speed_cbar,
                          hovertemplate="bed %{z:.0f} m<br>speed %{surfacecolor:.1f} m/yr<extra></extra>")

    if show_ice_surface or speed_on == "surface":
        if spd is not None and speed_on == "surface":
            fig.add_trace(go.Surface(x=GX, y=GY, z=srf, surfacecolor=spd, colorscale="Viridis",
                                     cmin=0, cmax=smax, opacity=speed_opacity, colorbar=speed_cbar,
                                     name="Ice surface (coloured by speed)", showlegend=True,
                                     hovertemplate="speed %{surfacecolor:.1f} m/yr<extra></extra>"))
            fig.update_traces(selector=dict(name="Bedmap3 bed"), showscale=False)
        else:
            fig.add_trace(go.Surface(x=GX, y=GY, z=srf, opacity=0.25, showscale=False,
                                     colorscale=[[0, "#dfe9f3"], [1, "#ffffff"]],
                                     name="Bedmap3 ice surface", showlegend=True, hoverinfo="skip"))

    # ---- elevation contours draped on the ice surface (a bit less transparent than the surface)
    if surface_contours and (show_ice_surface or speed_on == "surface"):
        import matplotlib
        import matplotlib.pyplot as _plt
        lo = np.ceil(np.nanmin(srf) / surface_contours) * surface_contours
        hi = np.floor(np.nanmax(srf) / surface_contours) * surface_contours
        levels = np.arange(lo, hi + surface_contours / 2, surface_contours)
        if levels.size:
            _f, _ax = _plt.subplots()
            cs = _ax.contour(gx, gy, srf, levels=levels)
            _plt.close(_f)
            X, Y, Z, T = [], [], [], []
            for level, segs in zip(cs.levels, cs.allsegs):
                for seg in segs:
                    if len(seg) < 2:
                        continue
                    X += list(seg[:, 0]) + [None]
                    Y += list(seg[:, 1]) + [None]
                    Z += [level + 0.004 * zspan] * len(seg) + [None]   # lift just above the surface
                    T += [f"{level:.0f} m"] * len(seg) + [None]
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z, mode="lines", opacity=surface_contour_opacity,
                line=dict(color=surface_contour_color, width=2), text=T,
                name=f"Surface contours ({surface_contours:g} m)", showlegend=True,
                hovertemplate="ice surface %{text}<extra></extra>"))

    # ---- speed contours draped on the ice surface (optional)
    if spd is not None and speed_contours:
        import matplotlib
        matplotlib.use(matplotlib.get_backend())
        import matplotlib.pyplot as _plt
        _f, _ax = _plt.subplots()
        cs = _ax.contour(gx, gy, np.where(np.isfinite(spd), spd, np.nan), levels=sorted(speed_contours))
        _plt.close(_f)
        for level, segs in zip(cs.levels, cs.allsegs):
            first = True
            for seg in segs:
                if len(seg) < 2:
                    continue
                cz = np.asarray(bm3["surf_interp"]((seg[:, 0], seg[:, 1])), float) + 0.01 * zspan
                fig.add_trace(go.Scatter3d(
                    x=seg[:, 0], y=seg[:, 1], z=cz, mode="lines",
                    line=dict(color="black", width=2), name=f"{level:g} m/yr",
                    legendgroup=f"spd{level}", showlegend=first,
                    hovertemplate=f"{level:g} m/yr contour<extra></extra>"))
                first = False

    # ---- layers, coloured by age; radar picks in black/grey
    ages = sorted({a for _, n, *_ in lines if (a := _age_years(n)) is not None})
    cmap = {a: AGE_COLORS[i % len(AGE_COLORS)] for i, a in enumerate(ages)}
    shown = set()
    def _order(item):
        n = item[1].lower()
        return (0, 0) if n.startswith("surface") else (2, 0) if n.startswith("bed") else (1, _age_years(item[1]) or 9e9)
    for pst, name, xl, yl, zl in sorted(lines, key=_order):
        low = name.lower()
        if low.startswith("surface"):
            color, label, w = "black", "Radar surface pick", 2
        elif low.startswith("bed"):
            color, label, w = "#555555", "Radar bed pick", 3
        else:
            a = _age_years(name)
            color = cmap.get(a, "#999999")
            label = f"{a/1000:.2f} ka isochrone" if a else name
            w = 4
        fig.add_trace(go.Scatter3d(
            x=xl, y=yl, z=zl, mode="lines", line=dict(color=color, width=w),
            name=label, legendgroup=label, showlegend=label not in shown,
            hovertemplate=f"{pst}<br>{name}<br>z=%{{z:.0f}} m<extra></extra>"))
        shown.add(label)
        a = _age_years(name)
        if label_ages and a is not None:
            pts = [(xx, yy, zz) for xx, yy, zz in zip(xl, yl, zl) if zz is not None and np.isfinite(zz)]
            if pts:
                xe, ye, ze = pts[-1]
                fig.add_trace(go.Scatter3d(x=[xe], y=[ye], z=[ze], mode="text",
                                           text=[f"{a:,.0f} yr"], textfont=dict(color=color, size=12),
                                           showlegend=False, hoverinfo="skip", legendgroup=label))

    # ---- flowlines draped on the ice surface
    for i, (label, fl) in enumerate((flowlines or {}).items()):
        fx, fy = np.asarray(fl[0], float), np.asarray(fl[1], float)
        fz = np.asarray(bm3["surf_interp"]((fx, fy)), float) + 0.01 * zspan
        tt = np.asarray(fl[2], float) if len(fl) > 2 else None
        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz, mode="lines", line=dict(color="#d62728", width=5, dash="dash"),
            name=label, legendgroup="flowlines",
            customdata=tt, hovertemplate=(f"{label}<br>travel time %{{customdata:.0f}} yr<extra></extra>"
                                          if tt is not None else f"{label}<extra></extra>")))

    # ---- markers (Takahe, dip picks, ...)
    for label, (mx, my) in (markers or {}).items():
        mz = float(np.asarray(bm3["surf_interp"]((np.array([mx]), np.array([my])))).ravel()[0])
        fig.add_trace(go.Scatter3d(x=[mx], y=[my], z=[mz + 0.03 * zspan], mode="markers+text",
                                   marker=dict(size=6, color="red", symbol="diamond"),
                                   text=[label], textposition="top center", name=label))

    fig.update_layout(
        title=f"{title} ({vert_exag:g}x vertical exaggeration)",
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(xaxis_title="x EPSG:3031 (m)", yaxis_title="y EPSG:3031 (m)",
                   zaxis_title="Elevation (m)",
                   aspectmode="manual", aspectratio=dict(x=xr, y=xr * yspan / xspan, z=exag)))
    return fig
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
LABEL_FONT = "Arial Black, Arial, sans-serif"   # heavy face so labels read over the surfaces


def bold_font(size=13, color=None, family="Arial, sans-serif"):
    """A light bold: weight='bold' on plotly >= 5.23; on older plotly just the size (no Arial Black)."""
    import plotly
    f = dict(size=size, family=family)
    if color is not None:
        f["color"] = color
    if tuple(int(v) for v in plotly.__version__.split(".")[:2]) >= (5, 23):
        f["weight"] = "bold"
    return f


def _font(color, size, family=LABEL_FONT):
    """Bold label font. plotly >= 5.23: weight='bold'; older plotly: Arial Black."""
    import plotly
    new = tuple(int(v) for v in plotly.__version__.split(".")[:2]) >= (5, 23)
    f = dict(color=color, size=size, family=family if new else LABEL_FONT)
    if new:
        f["weight"] = "bold"
    return f



# Landmark export lettering -> ages (years). Edit here if the key changes.
AGE_KEY = {"D1": 592, "D2": 1185, "D3": 1778, "D4": 2224,
           "D5": 2760.0, "D6": 4260, "D7": 1800.0}

# D1 0.592±0.014, D2 1.185±0.025, D7 1.778±0.038, D4 2.224±0.045, D6 4.260±0.099 ka AGE KEY (NEW); Good to note that I have no clue where the ages for D5 and D7 came from lol...!


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
                   x_range=None,
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
    x_range            : (xmin, xmax) in EPSG:3031 m to crop the plot in x; either end may be None,
                         e.g. (None, -1.25e6). Crops the grid, layers and flowlines.
    """
    xlo, xhi = (x_range or (None, None))
    xlo = -np.inf if xlo is None else xlo
    xhi = np.inf if xhi is None else xhi
    psts = psts or list(Corrected_PST_list.keys())

    # ---- collect layer points (respecting optional limits)
    lines, xs, ys = [], [], []
    for pst in psts:
        for name, L in _layer_items(Corrected_PST_list[pst]):
            x = np.asarray(L["x_array"], float)
            y = np.asarray(L["y_array"], float)
            z = np.asarray(L["z_array"], float)
            keep = np.isfinite(x) & np.isfinite(y) & (x >= xlo) & (x <= xhi)
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
    gx = np.arange(max(np.nanmin(xa) - pad, xlo), min(np.nanmax(xa) + pad, xhi) + res, res)
    gx = gx[gx <= xhi]
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
        if x_range:
            fx = np.where((fx >= xlo) & (fx <= xhi), fx, np.nan)   # NaN breaks the line outside
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
                                   text=[label], textposition="top center", name=label,
                                   textfont=bold_font(14, "white")))

    fig.update_layout(
        meta=dict(z_per_ve=exag / vert_exag, vert_exag=vert_exag, title_base=title),
        title=f"{title} ({vert_exag:g}x vertical exaggeration)",
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(xaxis_title="x EPSG:3031 (m)", yaxis_title="y EPSG:3031 (m)",
                   zaxis_title="Elevation (m)",
                   aspectmode="manual", aspectratio=dict(x=xr, y=xr * yspan / xspan, z=exag)))
    add_surface_opacity_slider(fig)
    add_vertical_exaggeration_slider(fig)
    return fig


# --------------------------------------------------------------------------------------
# Post-processing helpers. They work on any figure built by plot_layers_3d (old or new
# versions), so they can also be called on a figure that already exists.

def _is_flowline_trace(t):
    name = (getattr(t, "name", None) or "").lower()
    grp = (getattr(t, "legendgroup", None) or "").lower()
    return "flow" in grp or "flow" in name


def _is_isochrone_trace(t):
    grp = (getattr(t, "legendgroup", None) or "").lower()
    name = (getattr(t, "name", None) or "").lower()
    return "isochrone" in grp or "isochrone" in name


def style_labels(fig, label_size=13):
    """White, bold text for marker labels (e.g. Mt. Takahe).
    Isochrone traces and their age labels are left exactly as plotted."""
    for t in fig.data:
        if t.type != "scatter3d" or not t.mode or "text" not in t.mode or _is_isochrone_trace(t):
            continue
        t.textfont = _font("white", label_size)
    return fig


def _draped_trace_ids(fig):
    """Index of the ice-surface Surface trace and of every contour line draped on it."""
    surf = [i for i, t in enumerate(fig.data)
            if t.type == "surface" and "ice surface" in (t.name or "").lower()]
    cont = [i for i, t in enumerate(fig.data)
            if t.type == "scatter3d" and t.mode == "lines" and (
                (t.name or "").lower().startswith("surface contours")
                or (t.legendgroup or "").startswith("spd"))]
    return surf, cont


def add_surface_opacity_slider(fig, steps=None, y=0.0):
    """Slider for ice-surface opacity; draped contours fade with it and vanish at 0."""
    surf, cont = _draped_trace_ids(fig)
    if not surf:
        return fig
    s0 = fig.data[surf[0]].opacity
    s0 = 0.25 if s0 is None else s0
    c0 = {i: (1.0 if fig.data[i].opacity is None else fig.data[i].opacity) for i in cont}
    steps = steps if steps is not None else [round(v, 2) for v in
                                             [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]]
    ids = surf + cont
    slider_steps = []
    for v in steps:
        k = v / s0 if s0 > 0 else (0 if v == 0 else 1)
        vals = [v] * len(surf) + [min(1.0, c0[i] * k) for i in cont]
        vis = [True] * len(surf) + [v > 0] * len(cont)
        slider_steps.append(dict(method="restyle", label=f"{v:g}",
                                 args=[{"opacity": vals, "visible": vis}, ids]))
    active = min(range(len(steps)), key=lambda j: abs(steps[j] - s0))
    _set_slider(fig, dict(name="opacity", active=active, steps=slider_steps, x=0.02, len=0.45, y=y,
                          currentvalue=dict(prefix="Ice surface opacity: "), pad=dict(t=10, b=10)))
    return fig


def _set_slider(fig, slider):
    """Add a slider, replacing any existing one with the same name."""
    keep = [sl for sl in (fig.layout.sliders or []) if sl.name != slider["name"]]
    fig.layout.sliders = list(keep) + [slider]   # assign (update_layout would merge steps by index)


def add_vertical_exaggeration_slider(fig, values=(1, 2, 5, 10, 25, 50), y=0.0):
    """Slider that changes the vertical exaggeration live; the title updates to say the current value."""
    meta = fig.layout.meta or {}
    if "z_per_ve" not in meta:
        return fig
    base_ve = meta["vert_exag"]
    values = sorted(set(values) | {base_ve})
    base_title = meta.get("title_base", "")
    steps = [dict(method="relayout", label=f"{v:g}\u00d7",
                  args=[{"scene.aspectratio.z": float(meta["z_per_ve"] * v),
                         "title.text": f"{base_title} ({v:g}\u00d7 vertical exaggeration)"}])
             for v in values]
    _set_slider(fig, dict(name="vert_exag", active=values.index(base_ve), steps=steps,
                          x=0.53, len=0.45, y=y, pad=dict(t=10, b=10),
                          currentvalue=dict(prefix="Vertical exaggeration: ")))
    return fig

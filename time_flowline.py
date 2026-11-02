# -*- coding: utf-8 -*-
"""
time_flowline.py — trace ice through a velocity field by TIME instead of distance.

Give it any start point and a total time:
  direction="backward": where was this ice T years ago?   (upstream)
  direction="forward" : where will this ice be in T years? (downstream)
Each step moves the point by v * dt (dt in years). Velocity errors are sampled per
realization to give a spread of end points.

  trace_through_time(...)   : the function (use from code)
  run_interactive(...)      : prompts in the terminal for point, time, direction, dt, realizations

Interpolators follow your build_interpolator convention: f((x, y)).
"""
import numpy as np
import plotly.graph_objects as go


def _as_interp(err, like):
    """Accept an error grid as an array: wrap it in an interpolator on the same grid as `like`
    (a RegularGridInterpolator such as fvx). Transposes if needed to match like.values."""
    if err is None or callable(err):
        return err
    from scipy.interpolate import RegularGridInterpolator as RGI
    err = np.asarray(err, float)
    if not hasattr(like, "grid"):
        raise TypeError("Error grid given as an array, but fvx is not a RegularGridInterpolator; "
                        "pass an interpolator for the errors instead.")
    shape = like.values.shape
    if err.shape != shape:
        if err.T.shape == shape:
            err = err.T
        else:
            raise ValueError(f"Error grid shape {err.shape} does not match velocity grid {shape}.")
    return RGI(like.grid, err, bounds_error=False, fill_value=np.nan)


def trace_through_time(x0, y0, total_yr, fvx, fvy, ferrvx=None, ferrvy=None, *,
                       direction="backward", dt=5.0, n_real=1, err_mode="systematic",
                       z_interp=None, seed=None, Show_real_P_Bar=False, Show_step_P_bar = True):
    """
    x0, y0     : start point(s) in EPSG:3031 m (scalar or array, one per realization if arrays)
    total_yr   : total time to trace (years)
    direction  : "backward" (upstream, where was it) or "forward" (downstream, where will it go)
    fvx, fvy   : velocity interpolators f((x, y)) in m/yr
    ferrvx/y   : velocity-error interpolators (m/yr); None = no perturbation
    dt         : time step in years (0.5 = six months; 5-10 is usually plenty)
    n_real     : number of realizations
    err_mode   : "systematic" -> one random error factor per realization (bias that persists
                 along the path; the realistic case for mosaic velocity errors)
                 "random"     -> new random error every step (largely averages out)
    z_interp   : optional f((x, y)) for surface elevation along the path (for 3D plotting)
    Returns a list of dicts with x, y, z, t (years from the start point) and 'complete'
    (False if the trace left the velocity data before reaching total_yr). The end point is
    stored as x_end / y_end.
    """
    from tqdm import tqdm
    ferrvx = _as_interp(ferrvx, fvx); ferrvy = _as_interp(ferrvy, fvy)
    if direction not in ("backward", "forward"):
        raise ValueError("direction must be 'backward' or 'forward'")
    sgn = -1.0 if direction == "backward" else 1.0
    rng = np.random.default_rng(seed)
    x0 = np.broadcast_to(np.atleast_1d(np.asarray(x0, float)), (n_real,))
    y0 = np.broadcast_to(np.atleast_1d(np.asarray(y0, float)), (n_real,))
    nsteps = int(np.ceil(total_yr / dt))
    out = []
    loop = tqdm(range(n_real)) if Show_real_P_Bar else range(n_real)
    for r in loop:
        ax, ay = (rng.uniform(-1, 1, 2) if err_mode == "systematic" else (0.0, 0.0))

        def vel(px, py):
            u = float(fvx((px, py))); v = float(fvy((px, py)))
            if ferrvx is not None and ferrvy is not None:
                ex = float(ferrvx((px, py))); ey = float(ferrvy((px, py)))   # x error uses ERRX
                if err_mode == "random":
                    u += rng.uniform(-ex, ex); v += rng.uniform(-ey, ey)
                else:
                    u += ax * ex; v += ay * ey
            return u, v

        xs, ys, ts = [x0[r]], [y0[r]], [0.0]
        complete = True
        loop2 = tqdm(range(nsteps)) if Show_step_P_bar else range(nsteps)
        for k in loop2:
            h = min(dt, total_yr - ts[-1])
            u1, v1 = vel(xs[-1], ys[-1])
            if not (np.isfinite(u1) and np.isfinite(v1)):
                complete = False; break
            xm, ym = xs[-1] + sgn * 0.5 * h * u1, ys[-1] + sgn * 0.5 * h * v1   # RK2 midpoint
            u2, v2 = vel(xm, ym)
            if not (np.isfinite(u2) and np.isfinite(v2)):
                complete = False; break
            xs.append(xs[-1] + sgn * h * u2); ys.append(ys[-1] + sgn * h * v2); ts.append(ts[-1] + h)
        xs, ys, ts = np.array(xs), np.array(ys), np.array(ts)
        zs = np.asarray(z_interp((xs, ys)), float) if z_interp is not None else np.zeros_like(xs)
        out.append(dict(x=xs, y=ys, z=zs, t=ts, complete=complete, direction=direction,
                        x_end=xs[-1], y_end=ys[-1], years_traced=ts[-1],
                        distance_m=float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))))
    return out


def summarize(traces):
    """Mean end point, spread (1-sigma, m) and how many traces reached the full time."""
    xo = np.array([t["x_end"] for t in traces]); yo = np.array([t["y_end"] for t in traces])
    return dict(x_end=xo.mean(), y_end=yo.mean(),
                spread_m=float(np.sqrt(xo.var() + yo.var())),
                mean_distance_km=float(np.mean([t["distance_m"] for t in traces])) / 1e3,
                n_complete=int(sum(t["complete"] for t in traces)), n=len(traces))


def add_traces_to_fig(fig, traces, label, tick_every_yr=100, color_by_time=True, z_offset=50.0):
    """Draw realizations on an existing Plotly 3D figure, coloured by years from the start,
    with markers every `tick_every_yr` years and the end point highlighted."""
    for i, tr in enumerate(traces):
        fig.add_trace(go.Scatter3d(
            x=tr["x"], y=tr["y"], z=tr["z"] + z_offset, mode="lines",
            line=dict(width=4, color=tr["t"] if color_by_time else "red",
                      colorscale="Viridis", cmin=0, cmax=tr["t"].max() or 1,
                      showscale=(i == 0),
                      colorbar=dict(title="years ago" if tr["direction"] == "backward" else "years ahead",
                                    x=1.08, len=0.5)),
            name=label if i == 0 else None, showlegend=(i == 0), legendgroup=label,
            customdata=tr["t"], hovertemplate=f"{label}<br>%{{customdata:.0f}} yr<extra></extra>"))
        if tick_every_yr:
            idx = np.searchsorted(tr["t"], np.arange(tick_every_yr, tr["t"].max() + 1e-9, tick_every_yr))
            idx = idx[idx < tr["t"].size]
            fig.add_trace(go.Scatter3d(x=tr["x"][idx], y=tr["y"][idx], z=tr["z"][idx] + z_offset,
                                       mode="markers", marker=dict(size=2, color="black"),
                                       showlegend=False, legendgroup=label, hoverinfo="skip"))
    s = summarize(traces)
    zo = float(np.mean([t["z"][-1] for t in traces])) + z_offset
    end = "origin" if traces[0]["direction"] == "backward" else "destination"
    fig.add_trace(go.Scatter3d(x=[s["x_end"]], y=[s["y_end"]], z=[zo], mode="markers+text",
                               marker=dict(size=6, color="gold", line=dict(color="black", width=1)),
                               text=[f"{label}: {end}"], textposition="top center",
                               name=f"{label} {end}", legendgroup=label, showlegend=False))
    return fig


def _ask(prompt, default, cast=float):
    s = input(f"{prompt} [{default}]: ").strip()
    return default if s == "" else cast(s)


def run_interactive(fvx, fvy, ferrvx=None, ferrvy=None, z_interp=None, fig=None):
    """Terminal loop. Coordinates in EPSG:3031 metres (append 'km' to type kilometres).
    Returns a list of (inputs, traces) for every run. Enter 'q' at the x prompt to stop."""
    def _coord(v):
        v = v.strip().lower()
        return float(v[:-2]) * 1000 if v.endswith("km") else float(v)
    runs = []
    while True:
        sx = input("\nStart x (EPSG:3031 m, or e.g. -1368km; q to quit): ").strip()
        if sx.lower() in ("q", "quit", ""):
            break
        try:
            x0 = _coord(sx); y0 = _coord(input("Start y: "))
            T = _ask("Total time (years)", 640.0)
            direction = _ask("Direction (backward/forward)", "backward", str).lower()
            direction = "forward" if direction.startswith("f") else "backward"
            dt = _ask("Time step (years)", 5.0)
            n = int(_ask("Realizations", 20 if ferrvx is not None else 1))
        except ValueError as e:
            print(f"  could not read that: {e}"); continue
        tr = trace_through_time(x0, y0, T, fvx, fvy, ferrvx, ferrvy, direction=direction,
                                dt=dt, n_real=n, z_interp=z_interp)
        s = summarize(tr)
        word = "came from" if direction == "backward" else "will reach"
        print(f"  Over {T:g} yr this ice {word} x={s['x_end']:.0f}, y={s['y_end']:.0f} "
              f"({s['mean_distance_km']:.1f} km along flow; spread {s['spread_m']:.0f} m; "
              f"{s['n_complete']}/{s['n']} reached the full time)")
        if s["n_complete"] < s["n"]:
            print("  WARNING: some traces left the velocity data early; their end points are not valid.")
        if fig is not None:
            add_traces_to_fig(fig, tr, label=f"{direction} {T:g} yr from ({x0/1e3:.0f}, {y0/1e3:.0f}) km")
        runs.append((dict(x0=x0, y0=y0, T=T, direction=direction, dt=dt, n=n), tr))
    if fig is not None and runs:
        fig.show()
    return runs
# -*- coding: utf-8 -*-
"""
strain_tools.py — surface strain rates with uncertainty, vertical strain, and layer back-tracking.

1) strain_rates(...)      : exx, eyy, exy, flow-aligned longitudinal / lateral / shear, and
                            vertical strain ezz = -(dvx/dx + dvy/dy)  (rotation-invariant: no flow
                            direction needed). Optional Gaussian smoothing over ~1-4 ice thicknesses
                            (standard practice; surface strain is only meaningful at that scale).
2) strain_uncertainty(...): Monte Carlo — perturb vx, vy with their error fields (ERRX/ERRY from
                            MEaSUREs, or ITS_LIVE v_err) and recompute. Returns std of each strain
                            component, so you can mask |signal| < 2 sigma.
3) backtrack(...)         : trace a point upstream for `age` years through the velocity field ->
                            where that ice was when the layer was snow at the surface. Also returns
                            the cumulative vertical strain along the path (thickening factor).

Grid convention: vx, vy, ex, ey are 2D arrays shaped (ny, nx) on 1D axes x (nx) and y (ny),
both ascending. Units m and m/yr -> strain rates in 1/yr.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator as RGI


def _nan_gauss(a, sigma_px):
    """NaN-aware Gaussian smoothing."""
    if sigma_px <= 0:
        return a
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sigma_px)
    den = gaussian_filter(m.astype(float), sigma_px)
    out = num / np.where(den > 1e-6, den, np.nan)
    out[~m] = np.nan
    return out


def strain_rates(vx, vy, x, y, smooth_m=500.0):
    x, y = _axes(x, y, np.shape(vx))
    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    s = smooth_m / abs(dx) if smooth_m else 0.0
    u, v = _nan_gauss(vx, s), _nan_gauss(vy, s)
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    exx, eyy, exy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    sp = np.hypot(u, v)
    c, sn = u / np.where(sp > 0, sp, np.nan), v / np.where(sp > 0, sp, np.nan)
    lon = exx * c**2 + 2 * exy * c * sn + eyy * sn**2           # along flow
    lat = exx * sn**2 - 2 * exy * c * sn + eyy * c**2           # across flow (+ = extension)
    shr = (eyy - exx) * c * sn + exy * (c**2 - sn**2)           # in-plane shear
    ezz = -(exx + eyy)                                           # + = thickening (layers pushed up)
    return dict(exx=exx, eyy=eyy, exy=exy, lon=lon, lat=lat, shear=shr, ezz=ezz, speed=sp)


def strain_uncertainty(vx, vy, ex, ey, x, y, smooth_m=0.0, n=100, seed=0):
    """Std of each strain component from n noisy realisations. Noise is spatially uncorrelated
    per pixel, which is the worst case for derivatives (real errors are partly correlated, so
    this is a conservative upper bound)."""
    rng = np.random.default_rng(seed)
    keys = ["lon", "lat", "shear", "ezz"]
    acc = {k: [] for k in keys}
    for _ in range(n):
        r = strain_rates(vx + rng.normal(0, 1, vx.shape) * ex,
                         vy + rng.normal(0, 1, vy.shape) * ey, x, y, smooth_m)
        for k in keys:
            acc[k].append(r[k])
    return {k: np.nanstd(np.stack(v), axis=0) for k, v in acc.items()}


def _axes(x, y, shape=None):
    """Return clean 1D axes. Accepts 1D axes or 2D meshgrids; checks they match `shape` (ny, nx)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim == 2:
        x = x[0, :] if np.ptp(x[0, :]) > 0 else x[:, 0]
    if y.ndim == 2:
        y = y[:, 0] if np.ptp(y[:, 0]) > 0 else y[0, :]
    for name, a in (("x", x), ("y", y)):
        d = np.diff(a)
        if a.ndim != 1 or a.size < 2 or not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError(f"{name} axis must be 1D and strictly increasing or decreasing "
                             f"(got shape {a.shape}). Pass the velocity grid's own axes, "
                             f"e.g. Mv2 x_filtered / y_filtered.")
    if shape is not None and (y.size, x.size) != tuple(shape):
        raise ValueError(f"Grid shape {tuple(shape)} does not match axes (ny={y.size}, nx={x.size}). "
                         f"Grids must be (ny, nx); try passing grid.T or swapping x/y.")
    return x, y


def make_interp(grid, x, y):
    """Interpolator f((yq, xq)) for a (ny, nx) grid. Axes may be ascending or descending."""
    x, y = _axes(x, y, np.shape(grid))
    g = np.asarray(grid, float)
    if x[1] < x[0]:
        x, g = x[::-1], g[:, ::-1]
    if y[1] < y[0]:
        y, g = y[::-1], g[::-1, :]
    return RGI((y, x), g, bounds_error=False, fill_value=np.nan)


def backtrack(x0, y0, age, fvx, fvy, dt=5.0, fezz=None):
    """Follow ice at (x0, y0) upstream for `age` years (steady-state flow assumed).
    fvx, fvy, fezz: interpolators taking ((y, x)) as built by make_interp.
    Returns dict(path_x, path_y, x_dep, y_dep, distance_m, thickening) where thickening =
    exp(integral of ezz dt): >1 means the column has been thickened since deposition."""
    xs, ys = [x0], [y0]
    cum = 0.0
    t = 0.0
    while t < age:
        h = min(dt, age - t)
        def vel(px, py):
            return float(fvx((py, px))), float(fvy((py, px)))
        u1, v1 = vel(xs[-1], ys[-1])
        if not (np.isfinite(u1) and np.isfinite(v1)):
            break
        # RK2 upstream step
        xm, ym = xs[-1] - 0.5 * h * u1, ys[-1] - 0.5 * h * v1
        u2, v2 = vel(xm, ym)
        if not (np.isfinite(u2) and np.isfinite(v2)):
            break
        xs.append(xs[-1] - h * u2); ys.append(ys[-1] - h * v2)
        if fezz is not None:
            e = float(fezz((ym, xm)))
            cum += (e if np.isfinite(e) else 0.0) * h
        t += h
    xs, ys = np.array(xs), np.array(ys)
    return dict(path_x=xs, path_y=ys, x_dep=xs[-1], y_dep=ys[-1], years=t,
                distance_m=float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))),
                thickening=float(np.exp(cum)))


def forward_flowline(x0, y0, fvx, fvy, dist_m=150e3, ds=500.0):
    """Downstream flowline by arc length (for maps / the 3D plot). Returns x, y, travel time (yr)."""
    xs, ys, ts = [x0], [y0], [0.0]
    for _ in range(int(dist_m / ds)):
        u, v = float(fvx((ys[-1], xs[-1]))), float(fvy((ys[-1], xs[-1])))
        sp = np.hypot(u, v)
        if not np.isfinite(sp) or sp < 1e-3:
            break
        xs.append(xs[-1] + ds * u / sp); ys.append(ys[-1] + ds * v / sp); ts.append(ts[-1] + ds / sp)
    return np.array(xs), np.array(ys), np.array(ts)


def strain_along_trace(trace, strain, x, y, components=("lon", "lat", "shear", "ezz"), plot=True):
    """
    Strain-rate history along a time-traced path, and the total strain it adds up to.

    trace   : one trace dict from time_flowline.trace_through_time (keys x, y, t), or the list
              it returns (then the first realization is used).
    strain  : dict from strain_rates(...) (use smoothed rates, e.g. smooth_m ~ 2 ice thicknesses)
    x, y    : 1D ascending axes of the strain grids (same as passed to strain_rates)

    For each component returns the rate (1/yr) at every step and the accumulated strain:
        log_strain  = integral of rate dt  (sum of rate * dt)
        stretch     = exp(log_strain)       (final length / initial length)
        percent     = (stretch - 1) * 100   (negative = shortening, positive = stretching)
    'shear' is reported as accumulated shear strain gamma = integral of 2 * rate dt (an angle-like
    measure, not a length change). 'ezz' > 0 means the column was thickened.
    Assumes surface strain represents the column and that flow was steady.
    """
    if isinstance(trace, (list, tuple)):
        trace = trace[0]
    px, py, t = np.asarray(trace["x"]), np.asarray(trace["y"]), np.asarray(trace["t"])
    dt = np.diff(t, prepend=t[0])            # years per step (first step 0)
    out = {"t": t, "x": px, "y": py}
    for c in components:
        f = make_interp(strain[c], x, y)
        rate = np.asarray(f(np.column_stack([py, px])), float)
        r0 = np.where(np.isfinite(rate), rate, 0.0)
        if c == "shear":
            gamma = np.cumsum(2.0 * r0 * dt)
            out[c] = dict(rate=rate, gamma=gamma, total_gamma=float(gamma[-1]))
        else:
            ls = np.cumsum(r0 * dt)
            out[c] = dict(rate=rate, log_strain=ls, stretch=np.exp(ls),
                          total_percent=float((np.exp(ls[-1]) - 1) * 100))
        out[c]["frac_valid"] = float(np.mean(np.isfinite(rate)))
    for c in components:
        d = out[c]
        if c == "shear":
            print(f"{c:6s}: accumulated shear strain gamma = {d['total_gamma']:.2f}  ({d['frac_valid']:.0%} of path had data)")
        else:
            print(f"{c:6s}: total {d['total_percent']:+.1f}%  (stretch x{np.exp(d['log_strain'][-1]):.2f}; {d['frac_valid']:.0%} of path had data)")
    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for c in components:
            axs[0].plot(t, out[c]["rate"], label=c)
            axs[1].plot(t, out[c]["gamma"] if c == "shear" else (out[c]["stretch"] - 1) * 100,
                        label=c + (" (gamma)" if c == "shear" else " (%)"))
        axs[0].axhline(0, c="k", lw=0.6); axs[1].axhline(0, c="k", lw=0.6)
        axs[0].set_ylabel("Strain rate (1/yr)"); axs[1].set_ylabel("Accumulated strain")
        axs[1].set_xlabel("Years along the trace (from the start point)")
        axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
        axs[0].grid(alpha=0.25); axs[1].grid(alpha=0.25); plt.tight_layout(); plt.show()
    return out
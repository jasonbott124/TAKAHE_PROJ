# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:54:17 2026

@author: Jason
"""

# -*- coding: utf-8 -*-
"""
smooth_strain.py — surface strain rates from a SMOOTHED MEaSUREs velocity field.

Drop-in replacement for vnv.Strain_Deliverer() in Layer_Analysis_Master:
    from smooth_strain import smoothed_strains
    Strains = smoothed_strains(Mv2_vel, smooth_m=3000)

Returns an object with the same names your plotting code already uses
(lon_strain, lat_strain, shear_strain, eff_strain, lon_interp, lat_interp,
shear_interp, eff_interp) plus ezz / ezz_interp (vertical strain) and speed.

Why smooth: np.gradient on the raw 450 m grid amplifies pixel noise. Surface strain is only
meaningful over ~1-4 ice thicknesses, so velocities are Gaussian-smoothed (NaN-aware) at that
scale BEFORE differentiating. Units: velocities in m/yr -> strain rates in 1/yr (not 1/s).
"""
import numpy as np
from types import SimpleNamespace
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator as RGI


def _nan_gauss(a, sigma_px):
    """Gaussian smoothing that ignores NaNs (gaps don't bleed in, and stay gaps)."""
    if sigma_px <= 0:
        return a
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sigma_px)
    den = gaussian_filter(m.astype(float), sigma_px)
    out = num / np.where(den > 1e-6, den, np.nan)
    out[~m] = np.nan
    return out


def smoothed_strains(Mv2_vel, smooth_m=600.0, method="linear"):
    """
    Mv2_vel  : dict from Mv2_upload (needs "x_filtered", "y_filtered", "vx", "vy"; arrays (ny, nx))
    smooth_m : Gaussian sigma in metres (~1-2 ice thicknesses; 0 = no smoothing)
    """
    x = np.asarray(Mv2_vel["x_filtered"], float)
    y = np.asarray(Mv2_vel["y_filtered"], float)
    vx = np.asarray(Mv2_vel["vx"], float).copy()
    vy = np.asarray(Mv2_vel["vy"], float).copy()
    bad = ~np.isfinite(vx) | ~np.isfinite(vy) | ((vx == 0) & (vy == 0))   # 0/0 = no data in MEaSUREs
    vx[bad] = np.nan
    vy[bad] = np.nan

    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))   # dy < 0 if y is descending: fine
    s = smooth_m / abs(dx) if smooth_m else 0.0
    u, v = _nan_gauss(vx, s), _nan_gauss(vy, s)

    # np.gradient on (ny, nx) arrays returns [d/dy (axis 0), d/dx (axis 1)]
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    exx, eyy, exy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)

    # rotate into the local flow direction (c, s = unit flow vector; works in every quadrant)
    sp = np.hypot(u, v)
    c = u / np.where(sp > 0, sp, np.nan)
    sn = v / np.where(sp > 0, sp, np.nan)
    lon = exx * c**2 + 2 * exy * c * sn + eyy * sn**2            # along flow
    lat = exx * sn**2 - 2 * exy * c * sn + eyy * c**2            # across flow (+ = extension)
    shr = (eyy - exx) * c * sn + exy * (c**2 - sn**2)            # in-plane shear
    ezz = -(exx + eyy)                                            # incompressibility; + = thickening
    eff = np.sqrt(0.5 * (exx**2 + eyy**2 + ezz**2) + exy**2)     # effective strain rate

    mk = lambda a: RGI((y, x), a, method=method, bounds_error=False, fill_value=np.nan)  # call f((Y, X))
    return SimpleNamespace(
        lon_strain=lon, lat_strain=lat, shear_strain=shr, eff_strain=eff, ezz=ezz, speed=sp,
        lon_interp=mk(lon), lat_interp=mk(lat), shear_interp=mk(shr), eff_interp=mk(eff),
        ezz_interp=mk(ezz), x=x, y=y, smooth_m=smooth_m,
    )


def compare(Mv2_vel, smooth_list=(0, 1500, 3000, 6000), comp="lat_strain",
            xlim=(-1.55e6, -1.25e6), ylim=(-6.5e5, -4.7e5), vlim=0.03):
    """Quick look: the same strain component at several smoothing scales, to pick smooth_m."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(smooth_list), figsize=(5 * len(smooth_list), 5), constrained_layout=True)
    for ax, sm in zip(np.atleast_1d(axs), smooth_list):
        S = smoothed_strains(Mv2_vel, sm)
        pcm = ax.pcolormesh(S.x, S.y, getattr(S, comp), cmap="seismic", vmin=-vlim, vmax=vlim, shading="auto")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.set_title(f"{comp}, σ = {sm / 1000:g} km")
    fig.colorbar(pcm, ax=axs, label="strain rate (1/yr)", orientation="horizontal", shrink=0.5)
    plt.show()
    return fig
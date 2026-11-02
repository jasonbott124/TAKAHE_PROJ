# -*- coding: utf-8 -*-
"""
nye_drawdown.py — two numbers for the talk, from data already in Layer_Analysis_Master.

1) predict_layers(...)  Nye model with basal melt (Fahnestock et al. 2001, note 21) along one line:
     where SHOULD each dated layer sit, given the ice thickness, an accumulation rate (RACMO or a
     constant) and YOUR basal melt (Corrected_PST_list[pst]["bmelt"], mm/yr)?
     -> predicted vs observed layer elevation, the offset (observed - predicted), and how much of the
        prediction is due to melt at all (melt vs no-melt). Shows young layers barely feel steady melt.
     Cut the plot where ice is fast with xlim=(start_m, end_m) along the line.

2) infill_heat(...)     thickening of a dated interval in the trough vs the flank
     -> excess thickness dh, duration dt, melt rate dh/dt, heat flux q = rho_i * L * dh/dt.

Depths are the corrected radar depths (depth_radar, m below surface; firn-corrected) from
correct_layers_to_radar_surface. All results are steady-state 1D; treat them as order-of-magnitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

AGE_KEY = {"D1": 592, "D2": 1185, "D7": 1778, "D4": 2224, "D6": 4260}   # years (Y67a re-dating)
RHO_I, L_F, SEC_YR = 917.0, 3.34e5, 3.156e7


# ------------------------------------------------------------------ Nye with melt
def depth_nye_melt(t, H, lam, m):
    """Depth (m) of a layer of age t (yr). H thickness (m), lam accumulation (m ice/yr), m melt (m/yr).
    Vertical velocity falls linearly from -lam at the surface to -m at the bed."""
    t, H, lam, m = (np.asarray(v, float) for v in (t, H, lam, m))
    m = np.minimum(m, 0.999 * lam)                    # model needs m < lam
    r = m / lam
    z = H * (np.exp(-(lam - m) * t / H) - r) / (1.0 - r)
    return H - z


def _layer_id(name):
    import re
    g = re.search(r"(?<![A-Za-z0-9])(D\d+)(?![0-9])", name, re.I)
    return g.group(1).upper() if g else None


def _on_base(base_xy, L, key):
    """Value of layer L[key] at each base point (nearest pick within 200 m, else NaN)."""
    x, y = np.asarray(L["x_array"], float), np.asarray(L["y_array"], float)
    v = np.asarray(L[key], float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    out = np.full(len(base_xy), np.nan)
    if ok.sum() == 0:
        return out
    d, i = cKDTree(np.column_stack([x[ok], y[ok]])).query(base_xy)
    out[d < 200] = v[ok][i[d < 200]]
    return out


def predict_layers(Corrected_PST_list, pst, FilesByPST, accum, melt_mm=None, ages=AGE_KEY,
                   xlim=None, axis="abs_distance", plot=True, title=None):
    """
    accum   : accumulation in m ICE per yr: a number, an array on the base points, or a callable
              f(x, y) (e.g. a RACMO interpolator). RACMO SMB is usually kg m-2 yr-1 (= mm w.e./yr):
              convert with  lam = smb / 917.
    melt_mm : basal melt (mm/yr) on the base points; default Corrected_PST_list[pst]["bmelt"].
    xlim    : (start, end) along `axis` in metres; keep only the slow-moving part of the line.
    Returns a dict with axis, H, lam, melt and, per layer, observed/predicted depth and offset.
    """
    P = Corrected_PST_list[pst]
    base = P[FilesByPST[pst].base_file]
    bx, by = np.asarray(base["x_array"], float), np.asarray(base["y_array"], float)
    s = np.asarray(base.get(axis, base.get("x_position")), float)
    bxy = np.column_stack([bx, by])

    bed = P[FilesByPST[pst].bed_file]
    H = _on_base(bxy, bed, "depth_radar")                   # radar ice thickness at each base point
    S = np.asarray(base.get("surf_bm", np.full(len(bx), np.nan)), float)

    if callable(accum):
        lam = np.asarray(accum(bx, by), float)
    else:
        lam = np.broadcast_to(np.asarray(accum, float), bx.shape).astype(float)
    m = np.asarray(P["bmelt"] if melt_mm is None else melt_mm, float) / 1000.0
    m = np.where(np.isfinite(m), np.maximum(m, 0), 0.0)

    keep = np.isfinite(s) & np.isfinite(H) & np.isfinite(lam)
    if xlim is not None:
        keep &= (s >= min(xlim)) & (s <= max(xlim))

    out = dict(axis=s[keep], H=H[keep], lam=lam[keep], melt=m[keep] * 1000, S=S[keep], layers={})
    for name, L in P.items():
        if not (isinstance(L, dict) and "depth_radar" in L):
            continue
        lid = _layer_id(name)
        if lid not in ages:
            continue
        t = ages[lid]
        d_obs = _on_base(bxy, L, "depth_radar")[keep]
        d_pred = depth_nye_melt(t, H[keep], lam[keep], m[keep])
        d_nomelt = depth_nye_melt(t, H[keep], lam[keep], 0.0)
        out["layers"][lid] = dict(age=t, d_obs=d_obs, d_pred=d_pred, d_nomelt=d_nomelt,
                                  offset=d_obs - d_pred, melt_effect=d_pred - d_nomelt)

    order = sorted(out["layers"], key=lambda k: out["layers"][k]["age"])
    print(f"{pst}: Nye-with-melt prediction ({keep.sum()} points)")
    print(f"  melt {np.nanmin(out['melt']):.1f}–{np.nanmax(out['melt']):.1f} mm/yr, "
          f"accumulation {np.nanmin(out['lam']):.2f}–{np.nanmax(out['lam']):.2f} m ice/yr")
    for k in order:
        r = out["layers"][k]
        print(f"  {k} ({r['age'] / 1000:.2f} ka): melt lowers it by at most {np.nanmax(r['melt_effect']):.1f} m; "
              f"observed - predicted: median {np.nanmedian(r['offset']):+.0f} m, "
              f"max {np.nanmax(r['offset']):+.0f} m")

    if plot:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
        xs = out["axis"] / 1e3
        cols = plt.cm.viridis(np.linspace(0, 0.9, max(len(order), 1)))
        for c, k in zip(cols, order):
            r = out["layers"][k]
            a1.plot(xs, out["S"] - r["d_obs"], color=c, lw=2, label=f"{r['age'] / 1000:.2f} ka observed")
            a1.plot(xs, out["S"] - r["d_pred"], color=c, lw=1.5, ls="--", label=f"{r['age'] / 1000:.2f} ka Nye + melt")
            a2.plot(xs, r["offset"], color=c, lw=2, label=f"{r['age'] / 1000:.2f} ka")
        a1.set_ylabel("Layer elevation (m)")
        a1.legend(ncol=2, fontsize=9)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_ylabel("Observed − predicted depth (m)\n(+ = deeper than melt can explain)")
        a2.set_xlabel(f"Distance along {pst} (km)")
        ax3 = a2.twinx()
        ax3.plot(xs, out["melt"], color="0.5", lw=1, alpha=0.7)
        ax3.set_ylabel("Basal melt used (mm/yr)", color="0.4")
        a1.set_title(title or f"{pst}: dated layers vs Nye model with calculated basal melt")
        for a in (a1, a2):
            a.grid(alpha=0.25)
        plt.show()
        out["fig"] = fig
    return out


# ------------------------------------------------------------------ infill -> heat
def infill_heat(Corrected_PST_list, pst, FilesByPST, young="D2", old="D7", trough=None, flank=None,
                ages=AGE_KEY, axis="abs_distance"):
    """
    Thickness of the young->old interval in the trough vs on the flank (windows along `axis`, metres).
    dh = h_trough - h_flank is the extra snow that filled the low while that interval was laid down,
    over dt = age_old - age_young. Melt needed to make that room: dh/dt; heat flux q = rho_i L dh/dt.
    Lower bound: the interval has thinned since deposition, and inflow partly refills the low.
    """
    P = Corrected_PST_list[pst]
    base = P[FilesByPST[pst].base_file]
    bxy = np.column_stack([np.asarray(base["x_array"], float), np.asarray(base["y_array"], float)])
    s = np.asarray(base.get(axis, base.get("x_position")), float)
    dep = {}
    for name, L in P.items():
        if isinstance(L, dict) and "depth_radar" in L and _layer_id(name) in (young, old):
            dep[_layer_id(name)] = _on_base(bxy, L, "depth_radar")
    h = dep[old] - dep[young]
    win = lambda w: (s >= min(w)) & (s <= max(w)) & np.isfinite(h)
    h_t, h_f = np.nanmean(h[win(trough)]), np.nanmean(h[win(flank)])
    dh = h_t - h_f
    dt = ages[old] - ages[young]
    rate = dh / dt                                   # m/yr
    q = RHO_I * L_F * rate / SEC_YR                  # W/m^2
    print(f"{pst} {young}-{old} ({ages[young] / 1000:.2f}-{ages[old] / 1000:.2f} ka): "
          f"trough {h_t:.0f} m vs flank {h_f:.0f} m  ->  ratio {h_t / h_f:.2f}x, excess {dh:.0f} m over {dt} yr")
    print(f"  melt needed >= {rate * 1000:.0f} mm/yr  ->  heat flux >= {q:.2f} W/m^2 "
          f"({q / 0.08:.0f}x a typical 80 mW/m^2 background)")
    return dict(h_trough=h_t, h_flank=h_f, ratio=h_t / h_f, dh=dh, dt=dt, melt_m_yr=rate, q_W_m2=q)

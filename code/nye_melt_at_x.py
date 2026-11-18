# -*- coding: utf-8 -*-
"""
nye_at_x.py — interactive Nye-with-melt fit at one along-track position.

Uses the physics in nye_melt_fit.py (keep both files in the TAKAHE code folder).
Called from the LAYER ANALYSIS MASTER SCRIPT:

    import nye_at_x as nx
    nx.run_interactive(PST_layers, FilesByPST, Interpolators, NyeConfig)

For each x you type it prints the same fit results as nye_melt_fit.py
(DJ: lam, h, misfit; Nye-with-melt: lam, m, misfit; use_melt; bed age) plus
surface speed (Mv2), dataset accumulation, and a per-layer age table.

DATA LAYOUT (matches Data.Corrected_PST_list):
  - PST_layers[pst] = {layer_file_name: {'x_array','y_array','z_array','abs_distance',
    'surf_bm','bed_bm',...}, 'bmelt': ..., 'W': ...}. PST-level arrays are ignored.
  - along-track = 'abs_distance' (m), elevation = 'z_array', position = 'x_array'/'y_array'.
  - surface/bed = the layers whose names contain 'Surface' / 'Bed' (or FilesByPST picks).
  - ages parsed from names like '0,64ka_...' or '4,32_...' (override with NyeConfig.layer_ages).
  - depth and H are taken from the radar surface/bed picks, so any constant elevation
    offset in the correction cancels out.
"""
import re
import numpy as np
import pandas as pd

import Nye_Melt as nye

# candidate column names tried in order (first match wins) unless NyeConfig sets one
_CANDIDATES = {
    "x_col":    ["abs_distance", "absdist", "abs_dist", "dist_abs", "distance", "dist", "along_track"],
    "elev_col": ["z_array", "elevation", "abs_elev", "elev", "z_abs", "elev_abs", "z"],
    "xcoord":   ["x_array", "x", "x_ps", "ps_x", "x3031", "easting"],
    "ycoord":   ["y_array", "y", "y_ps", "ps_y", "y3031", "northing"],
}


def normalize_layers(pst_dict):
    """Corrected_PST_list[pst] -> {layer_name: DataFrame}.
    Keeps entries that are dicts of equal-length 1D arrays (the picked layers) and drops
    PST-level arrays such as 'bmelt', 'bmelt_all', 'W', 'X', 'Z', 'D'."""
    out = {}
    for name, val in pst_dict.items():
        if isinstance(val, pd.DataFrame):
            out[name] = val
        elif isinstance(val, dict):
            cols = {k: np.asarray(v) for k, v in val.items()
                    if np.ndim(v) == 1}
            lens = {len(v) for v in cols.values()}
            if cols and len(lens) == 1:
                out[name] = pd.DataFrame(cols)
    return out


def _find_surface_bed(layers, picks):
    """Surface/bed layer names: from FilesByPST if it has them, else by name."""
    surf, bed = _get(picks, "surface"), _get(picks, "bed")
    if surf not in layers:
        surf = next((n for n in layers if "surf" in n.lower()), None)
    if bed not in layers:
        bed = next((n for n in layers if re.search(r"(^|[^a-z])bed", n.lower())), None)
    return surf, bed


# ---------------------------------------------------------------- helpers
def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _find_col(df, cfg_value, kind):
    if cfg_value:
        if cfg_value not in df.columns:
            raise KeyError(f"NyeConfig.{kind}='{cfg_value}' not in columns: {list(df.columns)}")
        return cfg_value
    lower = {c.lower(): c for c in df.columns}
    for cand in _CANDIDATES[kind]:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"Could not auto-detect '{kind}'. Set NyeConfig.{kind}. Columns: {list(df.columns)}")


def _value_at(df, xq, xc, vc, max_gap):
    """Linear interpolation of column vc at along-track xq; NaN if xq is outside the
    pick or the nearest pick point is farther than max_gap (a gap in the layer)."""
    d = df[[xc, vc]].dropna().sort_values(xc)
    if d.empty:
        return np.nan
    xs, vs = d[xc].to_numpy(float), d[vc].to_numpy(float)
    if xq < xs[0] or xq > xs[-1] or np.min(np.abs(xs - xq)) > max_gap:
        return np.nan
    return float(np.interp(xq, xs, vs))


def _age_for(layer_name, ages):
    """Age in years. NyeConfig.layer_ages {substring: age_yr} wins; otherwise parse names
    like '0,64ka_...', '4,32_...', '1.80ka' as thousands of years."""
    for key, age in (ages or {}).items():
        if key.lower() in layer_name.lower():
            return float(age)
    m = re.match(r"\s*(\d+)[,.](\d+)\s*(ka)?", layer_name, re.I)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}") * 1000.0
    return None


def _interp_point(f, x, y):
    if f is None or not np.isfinite(x) or not np.isfinite(y):
        return np.nan
    try:
        return float(np.asarray(f(np.array([[x, y]])), float).ravel()[0])
    except Exception:
        return np.nan


# ---------------------------------------------------------------- extraction
def extract_column(layers, picks, xq, cfg, surf_interp=None, bed_interp=None):
    """Return dict with H, layer depths/ages (ice-equivalent) and x/y at along-track xq."""
    layers = normalize_layers(layers)
    names = list(layers.keys())
    surf_name, bed_name = _find_surface_bed(layers, picks)

    ref = layers[surf_name] if surf_name in layers else layers[names[0]]
    xc = _find_col(ref, cfg.x_col, "x_col")
    ec = _find_col(ref, cfg.elev_col, "elev_col")
    pxc = _find_col(ref, cfg.xcoord, "xcoord")
    pyc = _find_col(ref, cfg.ycoord, "ycoord")

    px = _value_at(ref, xq, xc, pxc, np.inf)
    py = _value_at(ref, xq, xc, pyc, np.inf)

    # surface / bed: radar picks first, Bedmap3 as fallback
    s = _value_at(layers[surf_name], xq, xc, ec, cfg.max_gap) if surf_name in layers else np.nan
    b = _value_at(layers[bed_name], xq, xc, ec, cfg.max_gap) if bed_name in layers else np.nan
    src = "radar picks"
    if cfg.surf_bed_source == "bedmap3" or not (np.isfinite(s) and np.isfinite(b)):
        s, b, src = _interp_point(surf_interp, px, py), _interp_point(bed_interp, px, py), "Bedmap3"

    rows = []
    for name, df in layers.items():
        if name in (surf_name, bed_name):
            continue
        age = _age_for(name, cfg.layer_ages)
        if age is None or age > cfg.max_age:
            continue
        z = _value_at(df, xq, xc, ec, cfg.max_gap)
        if np.isfinite(z):
            rows.append((name, age, s - z - cfg.firn_air_m))
    rows.sort(key=lambda r: r[1])

    # diagnostic: radar surface pick vs Bedmap3 surface stored alongside the picks
    surf_bm = np.nan
    if surf_name in layers and "surf_bm" in layers[surf_name].columns:
        surf_bm = _value_at(layers[surf_name], xq, xc, "surf_bm", cfg.max_gap)
    elif np.isfinite(px):
        surf_bm = _interp_point(surf_interp, px, py)

    return dict(x_along=xq, x=px, y=py, surf=s, bed=b, src=src, surf_bm=surf_bm,
                H=(s - b) - cfg.firn_air_m, layers=rows,
                x_range=(float(ref[xc].min()), float(ref[xc].max())), x_col=xc)


# ---------------------------------------------------------------- one point
def nye_at_x(layers, picks, interps, xq, cfg, plot=None):
    col = extract_column(layers, picks, xq, cfg,
                         _get(interps, "real_surf_interp"), _get(interps, "real_bed_interp"))
    H, rows = col["H"], col["layers"]

    speed = _interp_point(_get(interps, "Vel"), col["x"], col["y"])
    acc_raw = _interp_point(_get(interps, "Accumulation"), col["x"], col["y"])
    acc_ice = acc_raw * cfg.acc_to_m_ice if np.isfinite(acc_raw) else np.nan

    print("\n" + "=" * 64)
    print(f"{col['x_col']} = {xq:g}   (EPSG:3031 x={col['x']:.0f}, y={col['y']:.0f})")
    print(f"  surface/bed source  {col['src']}")
    print(f"  ice thickness H     {H:.1f} m (ice-eq, firn air {cfg.firn_air_m} m removed)")
    off = col["surf"] - col["surf_bm"]
    if np.isfinite(off):
        warn = "  <-- CHECK elevation correction" if abs(off) > 30 else ""
        print(f"  surface pick - BM3  {off:+.1f} m{warn}")
    flag = "OK (near-stagnant)" if speed <= cfg.max_speed else f"WARNING > {cfg.max_speed} m/yr: 1D steady-state assumption is weak"
    print(f"  surface speed Mv2   {speed:.2f} m/yr   -> {flag}")
    print(f"  dataset accum.      {acc_ice:.3f} m ice/yr (raw {acc_raw:.4g} x {cfg.acc_to_m_ice:g})")

    if not np.isfinite(H) or H <= 0:
        print("  No valid ice thickness here — skipping fit.")
        return None
    if len(rows) < 3:
        print(f"  Only {len(rows)} dated layers present at this x (need >= 3) — skipping fit.")
        for r in rows:
            print(f"    {r[0]:30s} age {r[1]:7.0f} yr  depth {r[2]:7.1f} m")
        return None

    names = [r[0] for r in rows]
    t = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])

    lam0 = acc_ice if np.isfinite(acc_ice) and acc_ice > 0 else 0.3
    out = nye.fit_trace(H, d, t, lam0=lam0)

    print("\n  Dansgaard-Johnsen fit")
    print(f"    dj_lam       {out['dj_lam']:.4g} m/yr")
    print(f"    dj_h         {out['dj_h']:.4g} m")
    print(f"    dj_misfit    {out['dj_misfit']:.4g}")
    print("  Nye-with-melt fit")
    print(f"    melt_lam     {out['melt_lam']:.4g} m/yr")
    print(f"    melt_m       {out['melt_m']:.4g} m/yr  ({out['melt_m'] * 1000:.1f} mm/yr)")
    print(f"    melt_misfit  {out['melt_misfit']:.4g}")
    print(f"    use_melt     {out['use_melt']}")
    tb = nye.bed_age_nye_melt(H, out["melt_lam"], out["melt_m"])
    print(f"    bed age      {tb:.0f} yr" if np.isfinite(tb) else "    bed age      infinite (no melt)")

    t_dj = nye.age_dj(d, H, out["dj_lam"], max(out["dj_h"], 0.0))
    t_m = nye.age_nye_melt(d, H, out["melt_lam"], out["melt_m"])
    print("\n  layer                          age_obs  depth   age_DJ  age_melt")
    for n, ti, di, a1, a2 in zip(names, t, d, t_dj, t_m):
        print(f"    {n[:28]:28s} {ti:7.0f} {di:7.1f} {a1:8.0f} {a2:9.0f}")

    out.update(dict(x_along=xq, x=col["x"], y=col["y"], H=H, surface_speed=speed,
                    dataset_accum=acc_ice, bed_age=tb, n_layers=len(rows)))

    if plot if plot is not None else cfg.plot:
        _plot(H, d, t, out, xq)
    return out


def _plot(H, d, t, out, xq):
    import matplotlib.pyplot as plt
    dd = np.linspace(0, H * 0.999, 400)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(t, d, "ko", label="dated layers")
    ax.plot(nye.age_dj(dd, H, out["dj_lam"], max(out["dj_h"], 0.0)), dd, "--",
            label=f"DJ  λ={out['dj_lam']:.3f}, h={max(out['dj_h'], 0):.0f} m")
    ax.plot(nye.age_nye_melt(dd, H, out["melt_lam"], out["melt_m"]), dd, "-",
            label=f"Nye+melt  λ={out['melt_lam']:.3f}, ṁ={out['melt_m']*1000:.0f} mm/yr")
    ax.axhline(H, color="0.4", lw=1)
    ax.set_xlim(0, max(t) * 1.6)
    ax.set_ylim(H * 1.02, 0)
    ax.set_xlabel("Age (yr)")
    ax.set_ylabel("Ice-equivalent depth (m)")
    ax.set_title(f"x = {xq:g}   speed = {out.get('surface_speed', np.nan):.1f} m/yr")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------- terminal loop
def run_interactive(PST_layers, FilesByPST, Interpolators, cfg):
    psts = list(PST_layers.keys())
    pst = psts[0]
    results = []
    norm = {p: normalize_layers(PST_layers[p]) for p in psts}
    for p in psts:
        s, b = _find_surface_bed(norm[p], FilesByPST.get(p) if isinstance(FilesByPST, dict) else None)
        dated = {n: _age_for(n, cfg.layer_ages) for n in norm[p] if n not in (s, b)}
        print(f"{p}: surface='{s}', bed='{b}'")
        for n, a in dated.items():
            print(f"   {n:35s} -> {'NO AGE (skipped)' if a is None else f'{a:.0f} yr'}")
    print(f"\nNye-with-melt point fit. PSTs: {psts}. Commands: a number = x (append 'km' for km), 'pst NAME', 'q'.")
    while True:
        layers = norm[pst]
        try:
            ref = next(iter(layers.values()))
            xc = _find_col(ref, cfg.x_col, "x_col")
            rng = (min(df[xc].min() for df in layers.values()), max(df[xc].max() for df in layers.values()))
            prompt = f"[{pst}] {xc} ({rng[0]:.4g} to {rng[1]:.4g}) > "
        except KeyError as e:
            print(e)
            return pd.DataFrame(results)
        s = input(prompt).strip()
        if s.lower() in ("q", "quit", "exit", ""):
            break
        m = re.match(r"pst\s+(\S+)", s, re.I)
        if m:
            pst = m.group(1) if m.group(1) in PST_layers else pst
            print(f"  active PST: {pst}")
            continue
        try:
            km = s.lower().endswith("km")
            xq = float(s[:-2] if km else s) * (1000.0 if km else 1.0)
        except ValueError:
            print("  enter a number (e.g. 195000 or 195km), 'pst NAME', or q")
            continue
        try:
            r = nye_at_x(layers, FilesByPST[pst], Interpolators, xq, cfg)
            if r:
                r["pst"] = pst
                results.append(r)
        except Exception as e:
            print(f"  error at x={xq}: {e}")
    df = pd.DataFrame(results)
    if not df.empty:
        print("\nSummary of points fitted this session:")
        print(df[["pst", "x_along", "H", "surface_speed", "dataset_accum",
                  "melt_lam", "melt_m", "melt_misfit", "dj_misfit", "use_melt", "bed_age"]].to_string(index=False))
    return df

def depth_nye_melt(t, H, lam, m):
    """Inverse: depth of a layer of age t. Useful for plotting the predicted layer."""
    r = m / lam
    z = H * (np.exp(-(lam - m) * np.asarray(t, float) / H) - r) / (1.0 - r)
    return H - z                                  # depth; > H means the layer has melted out
 
 
def bed_age_nye_melt(H, lam, m):
    """Age of the ice at the bed. Finite only when m > 0: older layers have been melted away."""
    return H / (lam - m) * np.log(lam / m) if m > 0 else np.inf
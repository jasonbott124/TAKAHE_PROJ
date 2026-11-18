"""
Fahnestock et al. (2001, Science 294:2338) layer-fitting approach, simplified.

For ONE radar trace (one ice column) you need:
  H      : ice thickness [m, ice-equivalent]
  d_i    : depths of dated layers below the surface [m, ice-equivalent]
  t_i    : ages of those layers [yr]

Step 1  Fit a Dansgaard-Johnsen (DJ) model  -> unknowns (accumulation lam, shear-layer height h)
Step 2  If best-fit h < 0 (thinning slower than accumulation can balance), refit with the
        Nye-with-melt model                 -> unknowns (accumulation lam, basal melt m)
Misfit (their note 23): F = sum_i [ (t_model(d_i) - t_i) / t_i ]^2

Depths MUST be ice-equivalent (remove the firn air column) or shallow layers will bias lam.
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- age-depth models
def age_nye_melt(d, H, lam, m):
    """Nye model with basal melt (their note 21). z = height above bed.
    Vertical velocity falls linearly from -lam at the surface to -m at the bed."""
    z = H - np.asarray(d, float)
    r = m / lam
    arg = r + (1.0 - r) * z / H
    with np.errstate(invalid="ignore", divide="ignore"):
        t = -H / (lam - m) * np.log(arg)
    return np.where(arg > 0, t, np.inf)


def age_dj(d, H, lam, h):
    """Dansgaard-Johnsen: uniform vertical strain above height h, linear decrease to zero below it.
    h = 0 reduces to plain Nye (sliding, no melt)."""
    z = H - np.asarray(d, float)
    if h <= 0:
        return H / lam * np.log(H / np.maximum(z, 1e-9))
    c = (2 * H - h) / (2 * lam)                 # = 1 / strain rate above h
    t_h = c * np.log((2 * H - h) / h)
    upper = c * np.log((2 * H - h) / np.maximum(2 * z - h, 1e-9))
    lower = t_h + (2 * H - h) / lam * (h / np.maximum(z, 1e-9) - 1.0)
    return np.where(z >= h, upper, lower)


def depth_nye_melt(t, H, lam, m):
    """Inverse: depth of a layer of age t. Useful for plotting the predicted layer."""
    r = m / lam
    z = H * (np.exp(-(lam - m) * np.asarray(t, float) / H) - r) / (1.0 - r)
    return H - z                                  # depth; > H means the layer has melted out


def bed_age_nye_melt(H, lam, m):
    """Age of the ice at the bed. Finite only when m > 0: older layers have been melted away."""
    return H / (lam - m) * np.log(lam / m) if m > 0 else np.inf


# ---------------------------------------------------------------- misfit + fitting
def misfit(t_model, t_obs):
    t_model = np.asarray(t_model, float)
    if not np.all(np.isfinite(t_model)):
        return 1e6
    return np.sum(((t_model - t_obs) / t_obs) ** 2)


def fit_trace(H, d, t, lam0=0.3, verbose=False):
    d, t = np.asarray(d, float), np.asarray(t, float)

    # Step 1: DJ fit over (lam, h); h allowed negative so the "h < 0" signal can appear
    def F_dj(p):
        lam, h = p
        if lam <= 0 or h >= H:
            return 1e6
        if h < 0:  # extend smoothly: treat as Nye plus a penalty growing with |h|
            return misfit(age_dj(d, H, lam, 0.0), t) * (1 + abs(h) / H)
        return misfit(age_dj(d, H, lam, h), t)

    dj = min((minimize(F_dj, [lam0, h0], method="Nelder-Mead") for h0 in (0.05 * H, 0.2 * H, 0.5 * H)),
             key=lambda r: r.fun)

    # Step 2: Nye-with-melt fit over (lam, m), with 0 <= m < lam
    def F_melt(p):
        lam, m = p
        if lam <= 0 or m < 0 or m >= lam:
            return 1e6
        return misfit(age_nye_melt(d, H, lam, m), t)

    melt = min((minimize(F_melt, [lam0, f * lam0], method="Nelder-Mead") for f in (0.01, 0.2, 0.5)),
               key=lambda r: r.fun)

    out = dict(dj_lam=dj.x[0], dj_h=dj.x[1], dj_misfit=dj.fun,
               melt_lam=melt.x[0], melt_m=melt.x[1], melt_misfit=melt.fun)
    # The paper's rule: use the melt model where DJ wants h <= 0 and melt fits better
    # (require a real improvement, not noise: here melt must halve the DJ misfit; tune to your picking error)
    out["use_melt"] = (dj.x[1] <= 0.01 * H) and (melt.fun < 0.5 * dj.fun)
    if verbose:
        for k, v in out.items():
            print(f"  {k:12s} {v:.4g}" if isinstance(v, float) else f"  {k:12s} {v}")
    return out


def misfit_grid(H, d, t, lams, ms):
    """Misfit surface over (lam, m). Plot it: the valley shows the trade-off and the uncertainty."""
    G = np.full((len(ms), len(lams)), np.nan)
    for i, m in enumerate(ms):
        for j, lam in enumerate(lams):
            if m < lam:
                G[i, j] = misfit(age_nye_melt(d, H, lam, m), t)
    return G


# ---------------------------------------------------------------- synthetic test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H = 1500.0                                   # m
    ages = np.array([640, 1000, 1500, 2000, 2500, 3000, 3600, 4320.0])

    for true_lam, true_m in [(0.35, 0.0), (0.35, 0.10), (0.35, 0.17)]:
        d = depth_nye_melt(ages, H, true_lam, true_m)
        d_obs = d + rng.normal(0, 5, d.size)     # 5 m picking error
        print(f"\nTRUE lam={true_lam} m/yr, m={true_m} m/yr; bed age = {bed_age_nye_melt(H, true_lam, true_m):.0f} yr")
        fit_trace(H, d_obs, ages, verbose=True)
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 08:49:40 2026

@author: Jason
"""

def statistical_melt(Arguments):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def get_arg(Arguments, name, default):
        if hasattr(Arguments, name):
            return getattr(Arguments, name)
        else:
            print(f"[INFO] '{name}' not provided. Using default = {default}")
            return default
    
    N        = get_arg(Arguments, "N", 1000)
    Thickness= get_arg(Arguments, "Thickness", np.array([1000.0]))
    surf_vel = get_arg(Arguments, "surf_vel", 100.0)     # m/yr
    theta    = get_arg(Arguments, "theta", 0.1)
    G        = get_arg(Arguments, "G", 60e-3)             # W/m^2
    L        = get_arg(Arguments, "L", 3.34e5)            # J/kg
    check_sizes = get_arg(Arguments, "check_sizes", False)
    m_mean  = get_arg(Arguments, "m_mean", 2.5)
    m_std   = get_arg(Arguments, "m_std", 0.5)
    n_mean  = get_arg(Arguments, "n_mean", 2.5)
    n_std   = get_arg(Arguments, "n_std", 0.5)
    dTdz_mean = get_arg(Arguments, "dTdz_mean", 0.04) 
    dTdz_std = get_arg(Arguments, "dTdz_mean", 0.02)
    k_mean = get_arg(Arguments, "k_mean", 0.04)
    k_std =  get_arg(Arguments, "k_std", 0.04)
    Bo_mean = get_arg(Arguments, "Bo_mean", 1.2e5)
    Bo_std = get_arg(Arguments, "Bo_std", 0.4e5)
    rho_mean = get_arg(Arguments, "rho_mean", 917)
    rho_std  = get_arg(Arguments, "rho_std", 8)
    plot_hists = get_arg(Arguments, "plot_hists", False) 
    bins = get_arg(Arguments, "bins", 100) 
    return_variables = get_arg(Arguments, "return_variables", False) 
    
    Bo_rng   = np.random.normal(loc = Bo_mean, scale = Bo_std/2, size=N)[:, None]       # (N,1) 
    rho_rng  = np.random.normal(loc = rho_mean,   scale = rho_std/2,      size=N)[:, None]       # (N,1)     # (your scale was 8e5/2; guessing typo)
    k_rng    = np.random.normal(loc = k_mean,   scale = k_std/2,  size=N)[:, None]       # (N,1)
    dTdz_rng = np.random.normal(loc = dTdz_mean,  scale = dTdz_std/2,   size=N)[:, None]       # (N,1)
    
    m = np.random.normal(loc= m_mean, scale = m_std, size=N)[:, None]       # (N,1) 
    n = np.random.normal(loc= n_mean, scale = n_std, size=N)[:, None]       # (N,1)
    
    H = np.asarray(Thickness)[None, :]
    
    # If surf_vel and theta are scalars, leave them.
    # If they’re arrays of length M, make them (1,M) like Thickness.
    sv = surf_vel if np.isscalar(surf_vel) else np.asarray(surf_vel)[None, :]
    th = theta    if np.isscalar(theta)    else np.asarray(theta)[None, :]
    
    tau_b = Bo_rng * (((m*n + 1) / (2*H)) * sv * (1 - th))**(1/n)  # (N,M)
    
    u_b = surf_vel * theta
    
    # Spatial arrays -> (1,M) if they’re per-point
    L2   = L if np.isscalar(L) else np.asarray(L)[None, :]   # avoid clobbering L name
    G2   = G if np.isscalar(G) else np.asarray(G)[None, :]
    ub2  = u_b if np.isscalar(u_b) else np.asarray(u_b)[None, :]
    
    if check_sizes == True:
        
        print('rho_rng:', np.shape(rho_rng))
        print('L2:', np.shape(L2))
        print('G2:', np.shape(G2))
        print('k_rng:', np.shape(k_rng))
        print('dTdz_rng:', np.shape(dTdz_rng))
        print('tau_b:', np.shape(tau_b))
        print('ub2:', np.shape(ub2))
    
    s2yr = 365.25*24*3600
    
    # If L is truly a scalar constant, you can just use it directly and skip L2.
    melt = (1 / (rho_rng * L2)) * (G2 - k_rng * dTdz_rng + (1/s2yr) * tau_b * ub2)   # -> (N,M)
    
    melt_in_mm = melt*s2yr*1000
    
    print('Melt Calculated')
    
    
    if plot_hists == True:
        
        # Flatten in case arrays are (N,1)
        Bo   = np.asarray(Bo_rng).ravel()
        rho  = np.asarray(rho_rng).ravel()
        k    = np.asarray(k_rng).ravel()
        dTdz = np.asarray(dTdz_rng).ravel()
        m    = np.asarray(m).ravel()
        n    = np.asarray(n).ravel()

        data = [
            (Bo,   r"$B_0$"),
            (rho,  r"$\rho$"),
            (k,    r"$k$"),
            (dTdz, r"$dT/dz$"),
            (m,    r"$m$"),
            (n,    r"$n$"),
        ]
        
        # --- Your palette ---
        palette = ["#D8DBE2", "#A9BCD0", "#58A4B0", "#373F51", "#1B1B1E"]
        LIGHT_FILL  = palette[1]   # A9BCD0 (light)
        LIGHT_EDGE  = palette[2]   # 58A4B0 (still light, good outline)
        TEXT_COLOR  = palette[0]   # D8DBE2 (very light, good on dark PPT background)
        
        """
        Expects arrays shaped (N,) or (N,1). Transparent fig/axes for dark PPT backgrounds.
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    
        # Transparent backgrounds
        fig.patch.set_alpha(0.0)
        for ax in axes.ravel():
            ax.set_facecolor("none")
    
            # Subtle styling for dark PPT background
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(TEXT_COLOR)
            ax.spines["bottom"].set_color(TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    
            # Light grid (optional, looks good on dark backgrounds)
            ax.grid(True, alpha=0.18, linewidth=0.8, color=TEXT_COLOR)
    
        for ax, (arr, label) in zip(axes.ravel(), data):
            ax.hist(arr, bins=bins, color=LIGHT_FILL, edgecolor=LIGHT_EDGE, linewidth=1.2, alpha=0.9)
            ax.set_title(label, color=TEXT_COLOR, fontsize=13, pad=8)
    
            # Mean line (light) for readability
            mu = np.mean(arr)
            ax.axvline(mu, color=palette[0], linewidth=2.0, alpha=0.9)
            ax.text(
                0.98, 0.92, rf"$\mu={mu:.3g}$",
                transform=ax.transAxes, ha="right", va="top",
                color=TEXT_COLOR, fontsize=10
            )
    
        # Overall title
        fig.suptitle('Melt Calculation Variables', color=TEXT_COLOR, fontsize=16, y=1.02)
    
        # # Save with true transparency (best for PowerPoint)
        # if savepath is not None:
        #     plt.savefig(savepath, dpi=dpi, transparent=True, bbox_inches="tight")
    
        plt.show()
        
        if return_variables == True:
            
            return melt_in_mm, tau_b, fig, data
        
        else:
            
            return melt_in_mm, tau_b, fig
    
    if return_variables == True:
        
        # Flatten in case arrays are (N,1)
        Bo   = np.asarray(Bo_rng).ravel()
        rho  = np.asarray(rho_rng).ravel()
        k    = np.asarray(k_rng).ravel()
        dTdz = np.asarray(dTdz_rng).ravel()
        m    = np.asarray(m).ravel()
        n    = np.asarray(n).ravel()

        data = [
            (Bo,   r"$B_0$"),
            (rho,  r"$\rho$"),
            (k,    r"$k$"),
            (dTdz, r"$dT/dz$"),
            (m,    r"$m$"),
            (n,    r"$n$"),
        ]
        
        return melt_in_mm, tau_b, data

    
    return melt_in_mm, tau_b

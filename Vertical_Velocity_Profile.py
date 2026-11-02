# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

def coords_to_abs_distance(x, y, *, start_at_zero=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy)

    s = np.empty(x.size, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(ds)

    if not start_at_zero:
        # if you want to preserve some existing offset, remove this
        pass

    return s


def vertical_velocity_profile(PST_layers, FilesByPST, Interpolators, ArgumentsByPST, nz=1000):
    """
    Compute W(z,x) on a sigma grid for each PST and store W, X, Z back into PST_layers[pst_name].

    Fixes:
      - Correct masking: mask applies along x (columns): W[:, mask]
      - Correct nondimensional coordinate: sigma = (Z - bed)/H (NOT Z/H)
    """

    sigma_1d = np.linspace(0.0, 1.0, nz)  # 0=bed, 1=surface

    for pst_name, PST in PST_layers.items():

        base_file = FilesByPST[pst_name].base_file
        x = np.asarray(PST[base_file]["x_array"], dtype=float)
        y = np.asarray(PST[base_file]["y_array"], dtype=float)

        bed      = np.asarray(Interpolators.real_bed_interp((x, y)), dtype=float)
        surf     = np.asarray(Interpolators.real_surf_interp((x, y)), dtype=float)
        adot     = np.asarray(Interpolators.Accumulation((x, y)), dtype=float)
        surf_vel = np.asarray(Interpolators.Vel((x, y)), dtype=float)

        n = float(ArgumentsByPST[pst_name].n)

        H = surf - bed
        # Avoid divide-by-zero or negative thickness
        H_safe = np.where(H > 0, H, np.nan)

        # grids (nz, nx)
        X = np.tile(x[None, :], (nz, 1))
        Z = bed[None, :] + sigma_1d[:, None] * H_safe[None, :]

        # nondimensional depth coordinate in [0,1]
        sigma2d = (Z - bed[None, :]) / H_safe[None, :]
        sigma2d = np.clip(sigma2d, 0.0, 1.0)

        Wshape = np.full_like(Z, np.nan, dtype=float)

        mask_fast = surf_vel > 20.0
        mask_slow = ~mask_fast

        # Fast regime
        Wshape[:, mask_fast] = -sigma2d[:, mask_fast]

        # Slow regime
        Wshape[:, mask_slow] = (
            (-(n + 2.0) / (n + 1.0)) * sigma2d[:, mask_slow]
            + (1.0 / (n + 1.0)) * (1.0 - (1.0 - sigma2d[:, mask_slow]) ** (n + 2.0))
        )
        
        # if hasattr()
        # Scale by accumulation
        W = (adot[None, :] * Wshape) - (PST_layers[pst_name]['bmelt'][None,:]*0.001)

        PST_layers[pst_name]["W"] = W
        PST_layers[pst_name]["X"] = X
        PST_layers[pst_name]["Z"] = Z

    return PST_layers


def vertical_velocity_plot(
    FilesByPST,
    Interpolators,
    ArgumentsByPST,
    Data,
    *,
    pst_list=None,
    nz=300,
    cmap="viridis",
    w_lims=None,
    save_dir=None,
    show=True,
    transparent=True,
):
    """
    Plot ONE PST per figure:
      - 2D colormap of vertical velocity W(z,x)
      - Uses Data.Corrected_PST_list (which is PST_layers in your framework)

    If W/X/Z are not already in PST_layers, it computes them first.
    """

    PST_layers = Data.Corrected_PST_list  # <-- this is your PST_layers dict

    if pst_list is None:
        pst_list = list(PST_layers.keys())

    # Compute W if missing for any PST in pst_list
    need_compute = any(
        ("W" not in PST_layers[p]) or ("X" not in PST_layers[p]) or ("Z" not in PST_layers[p])
        for p in pst_list
    )
    if need_compute:
        vertical_velocity_profile(PST_layers, FilesByPST, Interpolators, ArgumentsByPST, nz=nz)

    for pst_name in pst_list:
        
        base_file = FilesByPST[pst_name].base_file
        
        args = ArgumentsByPST[pst_name]
        # X = PST_layers[pst_name]["X"]
        Distance = PST_layers[pst_name][base_file]["abs_distance"]
        Z = PST_layers[pst_name]["Z"]   # (nz, nx)
        W = PST_layers[pst_name]["W"]
        
        # Make Distance into a (nz, nx) grid
        D = np.tile(Distance[None, :], (W.shape[0], 1)) 
        
        PST_layers[pst_name]["D"] = D
        
        # D,Z = np.meshgrid(Distance,z)

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

        pcm = ax.pcolormesh(D, Z, W, shading="auto", cmap=cmap)
        if w_lims is not None:
            pcm.set_clim(*w_lims)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Vertical velocity (w)")

        ax.set_title(f"{pst_name} — Vertical Velocity")
        ax.set_xlabel("Along-profile distance (x)")
        ax.set_ylabel("Elevation (z)")

        # Optional: match your layer plot window if those attrs exist
        if hasattr(args, "limits_abs") and args.limits_abs is not None:
            ax.set_xlim(args.limits_abs[0], args.limits_abs[1])
        if hasattr(args, "layer_ylim") and args.layer_ylim is not None:
            ax.set_ylim(args.layer_ylim[0], args.layer_ylim[1])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"{pst_name}_vertical_velocity.png")
            fig.savefig(out, dpi=300, transparent=transparent)
            

        if show:
            plt.show()
        else:
            plt.close(fig)

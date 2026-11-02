def layer_corr_plot_multiPST(
    FilesByPST,
    Interpolators,
    ArgumentsByPST,
    Data,
    Strains,
    *,
    x_limits_by_pst=None,          # dict like {"X47a": (0,120e3), ...} or None -> prompt
    axis_key="abs_distance",  # default per your request
    start_index=0,                 # keep at 0 unless you truly have junk entries
    global_vel_lims=None,
    global_strain_lims=None,
    global_melt_lims=None,
    enforce_constant_layer_scale=True,  # prints best-practice note
    scale = None,
    global_spec_lims = None,
    global_foc1_pow_lims = None,
    global_foc2_pow_lims = None,
    dataset_option = 1,
    radar_df = None
):
    """
    Multi-PST version of your frac_depth_plot:
      - Loops over Data.Corrected_PST_list (dict of PST -> layers dict)
      - Produces one "original-style" 4-stack plot per PST, displayed side-by-side
      - Uses absolute distance for the x-axis across velocity/strain/melt/layers
      - Uses global y-limits (vel/strain/melt) for easy comparison across PSTs
      - Handles per-PST x-limits (prompt if not provided)

    Requirements for each PST layers dict:
      each layer has "x_array","y_array", and axis_key (default "absolute_distance")
      z arrays:
        - fractional depth: layer["depth_frac"] or layer["frac_depth"] (we try both)
        - elevation: layer["z_abs"] or layer["z_array"] depending on your corrected structure

    FilesByPST:
      dict keyed by PST name. Each value must have attributes:
        .base_file .bed_file .surf_file

    ArgumentsByPST:
      dict keyed by PST name. Each value should provide:
        .depth ("Fractional" or "Elevation")
        .plot_style ("line" or "scatter")
        .point_size
        .low_thresh .upper_thresh .Bo .m .n .melt_mode
        (optional per PST)
        .limits  (ignored if x_limits_by_pst is provided; used otherwise)
        .vel_lims, .strain_lims, .melt_lims (optional; overridden by globals if provided)

    NOTE: This function assumes velocity, strains, and melt are computed from (x_array,y_array)
          but plotted against axis_key.
    """

    # def _plot_layers_for_pst(ax, pst_name, FilesByPST, Interpolators, ArgumentsByPST, Data,
    #                          *, draw_legend=False):
    #     """
    #     Internal helper: plots the EXACT radar profiles/layers that layer_corr_plot_multiPST uses.
    #     vertical_velocity_plot will import and reuse this so the overlays match perfectly.
    #     """
    #     args = ArgumentsByPST[pst_name]
    
    #     # ---- EVERYTHING BELOW should be copied from your existing plotting logic ----
    #     # Example placeholders (replace with your real code):
    #     #
    #     # layers_df = Data.Corrected_PST_list[pst_name]
    #     # limits = args.limits_abs   # or however you apply x-limits
    #     # ... plot bed/surface/base if you do ...
    #     # ... plot each layer line exactly the same as you currently do ...
    #     #
    #     # ax.plot(x, y, ...)
    #     # ax.set_xlim(limits)
    #     #
    #     # -------------------------------------------------------------
    
    #     if draw_legend:
    #         ax.legend()

    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------
    # Helpers
    # -------------------------
    
    def _apply_plot_scale(fig, base_figsize=(18, 18), scale = scale):
        """
        Scale fonts/ticks/lines with the figure.
        - base_figsize: the figsize you tuned your current params for
        - scale: override manual scale if you want (e.g., 0.8, 1.2)
        """
        w, h = fig.get_size_inches()
        bw, bh = base_figsize
    
        # "area-ish" scaling is usually nicer than only width or only height
        s_auto = np.sqrt((w * h) / (bw * bh))
        s = float(scale) if scale is not None else float(s_auto)
    
        # Clamp so things don’t get ridiculous
        s = np.clip(s, 0.6, 2.0)
    
        sizes = {
            "title": 18 * s,
            "label": 16 * s,
            "tick": 12 * s,
            "legend": 12 * s,
            "annot": 18 * s,
            "line": 1.1 * s,
            "grid": 0.6 * s,
            "marker": 10  * (s**2),  # marker area scales like s^2
        }
        return sizes
    
    def _require(cond, msg):
        if not cond:
            raise ValueError(msg)

    def _set_transparent(fig, axes):
        fig.patch.set_alpha(0)
        for a in axes:
            a.set_facecolor("none")

    def _compute_speed(Vel, x, y):
        vx = Vel['fvx']((x, y))
        vy = Vel['fvy']((x, y))
        return np.sqrt(vx * vx + vy * vy)

    def _get_layer_z(layer, mode):
        """mode: 'Fractional' or 'Elevation'"""
        if mode == "Fractional":
            # You have used both 'depth_frac' and 'frac_depth' in your code
            if "depth_frac" in layer:
                return np.asarray(layer["depth_frac"], dtype=float)
            if "frac_depth" in layer:
                return np.asarray(layer["frac_depth"], dtype=float)
            # fallback
            return np.asarray(layer.get("z_array", np.full_like(layer[axis_key], np.nan)), dtype=float)
        else:
            # elevation: prefer corrected output if present
            if "z_abs" in layer:
                return np.asarray(layer["z_abs"], dtype=float)
            return np.asarray(layer.get("z_array", np.full_like(layer[axis_key], np.nan)), dtype=float)

    def _mask_sort_line(x, y):
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.ma.masked_invalid(x), np.ma.masked_invalid(y)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        
        return np.ma.masked_invalid(x), np.ma.masked_invalid(y)

    def _prompt_limits(pst_name, axis_vals):
        
        amin = float(np.nanmin(axis_vals))
        amax = float(np.nanmax(axis_vals))
        print(f"\n[{pst_name}] axis '{axis_key}' range: [{amin:.3f}, {amax:.3f}]")
        raw = input(f"[{pst_name}] Enter x-limits (xmin xmax) in {axis_key} units: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"[{pst_name}] Could not parse two numbers for limits.")
        x0, x1 = float(parts[0]), float(parts[1])
        if x1 < x0:
            x0, x1 = x1, x0
        
        return (x0, x1)

    def _get_limits_for_pst(pst_name, base_axis_vals):
        if x_limits_by_pst is not None and pst_name in x_limits_by_pst:
            x0, x1 = x_limits_by_pst[pst_name]
            x0, x1 = float(x0), float(x1)
            if x1 < x0:
                x0, x1 = x1, x0
            return (x0, x1)
        return _prompt_limits(pst_name, base_axis_vals)

    # -------------------------
    # Validate + grab data
    # -------------------------
    _require(hasattr(Data, "Corrected_PST_list"), "Data.Corrected_PST_list missing")
    _require(hasattr(Data, "Vel"), "Data.Vel missing")

    Corrected_PST_list = Data.Corrected_PST_list
    Vel = Data.Vel

    pst_names = list(Corrected_PST_list.keys())
    if len(pst_names) == 0:
        print("No PSTs found in Data.Corrected_PST_list")
        return

    # Best-practice note
    if enforce_constant_layer_scale:
        print(
            "\nNOTE (best practice): For comparing folding between PSTs, it’s best to keep a constant\n"
            "vertical/horizontal sense of scale in the radargram layer panels. It’s not strictly required,\n"
            "but changing layer scales can make fold amplitude/wavelength comparisons misleading.\n"
        )

    # -------------------------
    # First pass: compute global limits if not provided
    # -------------------------
    all_speeds = []
    all_lat_strains = []
    all_melts = []

    for pst in pst_names:
        
        layers = Corrected_PST_list[pst]
        layer_items = list(layers.items())[start_index:]
        
        print("layers:", type(layers))
        print("layer_items:", type(layer_items))
        print('line 216')

        # pull base layer coordinate arrays for physics (x,y)
        Files = FilesByPST[pst]
        base = layers[Files.base_file]
        base_x = np.asarray(base["x_array"], dtype=float)
        base_y = np.asarray(base["y_array"], dtype=float)

        speed = _compute_speed(Vel, base_x, base_y)
        all_speeds.append(speed)

        lat = Strains.lat_interp((base_y, base_x))
        all_lat_strains.append(lat)

        # Melt: keep same model choice per PST
        Args = ArgumentsByPST[pst]
        melt_mode = getattr(Args, "melt_mode", "deterministic")

        B2D = Interpolators.real_bed_interp
        S2D = Interpolators.real_surf_interp
        thickness = S2D((base_x, base_y)) - B2D((base_x, base_y))

        theta = np.zeros_like(speed, dtype=float)
        mask = (speed > Args.low_thresh) & (speed < Args.upper_thresh)
        denom = (Args.upper_thresh - Args.low_thresh)
        if denom <= 0:
            raise ValueError(f"[{pst}] upper_thresh must be > low_thresh")
        theta[mask] = (speed[mask] - Args.low_thresh) / denom
        theta[speed >= Args.upper_thresh] = 1.0

        if melt_mode == "statistical":
            
            # If you want global melt limits based on statistical mean, compute mean
            import stat_melt as stm
            Args.Thickness = thickness
            Args.surf_vel = speed
            Args.theta = theta
            melt_mc_mm, _ = stm.statistical_melt(Args)
            melt_mean = np.nanmean(melt_mc_mm, axis=0)
            all_melts.append(melt_mean) # calculates melt
        
        else:
            tau_b = Args.Bo * (
                ((Args.m * Args.n + 1) / (2 * thickness)) * speed * (1 - theta)
            ) ** (1 / Args.n)
            s2yr = (60 ** 2) * 24 * 365.25
            G, dTdz, k, rho, L = 0.06, 0.02, 2.2, 917.0, 3.34e5
            u_b = speed * theta
            mrate = (1 / (rho * L)) * (G - k * dTdz + (1 / s2yr) * tau_b * u_b)
            melt_in_mm = mrate * s2yr * 1000
            all_melts.append(melt_in_mm)

    # Build global limits if not given
    if global_vel_lims is None:
        vcat = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in all_speeds if v is not None])
        global_vel_lims = (float(np.nanmin(vcat)), float(np.nanmax(vcat))) if vcat.size else (0.0, 1.0)

    if global_strain_lims is None:
        scat = np.concatenate([np.ravel(s[np.isfinite(s)]) for s in all_lat_strains if s is not None])
        # Use symmetric if you want; otherwise min/max
        if scat.size:
            sabs = float(np.nanmax(np.abs(scat)))
            global_strain_lims = (-sabs, sabs)
        else:
            global_strain_lims = (-0.05, 0.05)

    if global_melt_lims is None:
        mcat = np.concatenate([np.ravel(m[np.isfinite(m)]) for m in all_melts if m is not None])
        global_melt_lims = (float(np.nanmin(mcat)), float(np.nanmax(mcat))) if mcat.size else (0.0, 1.0)

    # -------------------------
    # Figure: one PST per column, each column has 4 stacked axes
    # -------------------------
    plt.style.use("dark_background")

    # nP = len(pst_names)
    # fig = plt.figure(figsize=(18 * nP, 18))
    # fig.set_dpi(150)
    # fig.patch.set_alpha(0)

    # # # grids: 4 rows x nP cols
    # gs = fig.add_gridspec(4, nP, height_ratios=[1, 1, 1, 2], hspace=0.08, wspace=0.12)
    
    fig_base_per_pst = (18, 18)  # your current "tuned" baseline per PST

    nP = len(pst_names)
    fig_w = fig_base_per_pst[0] * nP
    fig_h = fig_base_per_pst[1]
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, constrained_layout = True)
    sizes = _apply_plot_scale(fig, base_figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(4, nP, height_ratios=[1, 1, 1, 2])

    for j, pst in enumerate(pst_names):
        layers = Corrected_PST_list[pst]
        layer_items = list(layers.items())[start_index:]
        
        print("layers:", type(layers))
        print("layer_items:", type(layer_items))
        print('line 317')

        Files = FilesByPST[pst]
        Args = ArgumentsByPST[pst]

        _require(Files.base_file in layers, f"[{pst}] base_file not in layers: {Files.base_file}")
        _require(Files.bed_file in layers, f"[{pst}] bed_file not in layers: {Files.bed_file}")
        _require(Files.surf_file in layers, f"[{pst}] surf_file not in layers: {Files.surf_file}")
        _require(Args.depth in ("Fractional", "Elevation"), f"[{pst}] Arguments.depth invalid")

        base = layers[Files.base_file]
        base_x = np.asarray(base["x_array"], dtype=float)
        base_y = np.asarray(base["y_array"], dtype=float)
        base_abs = np.asarray(base["abs_distance"], dtype=float)

        # X-axis: absolute distance (per-point) from base layer
        base_axis = np.asarray(base.get(axis_key, np.full_like(base_abs, np.nan)), dtype=float)
        x0, x1 = _get_limits_for_pst(pst, base_axis)

        # mask for plotting inside limits
        base_m = np.isfinite(base_axis) & (base_axis >= x0) & (base_axis <= x1)

        # Create axes
        ax_vel    = fig.add_subplot(gs[0, j])
        ax_strain = fig.add_subplot(gs[1, j], sharex=ax_vel)
        ax_tau    = fig.add_subplot(gs[2, j], sharex=ax_vel)
        ax_layers = fig.add_subplot(gs[3, j], sharex=ax_vel)

        _set_transparent(fig, [ax_vel, ax_strain, ax_tau, ax_layers])
        
        if dataset_option == 1:

            # -------------------------
            # Velocity
            # -------------------------
            speed = _compute_speed(Vel, base_x, base_y)
            ax_vel.scatter(
                base_axis[base_m], speed[base_m],
                c=speed[base_m], cmap="viridis",
                vmin=global_vel_lims[0], vmax=global_vel_lims[1],
                s=max(5, getattr(Args, "point_size", 10))
            )
            ax_vel.set_ylim(global_vel_lims[0], global_vel_lims[1])
            ax_vel.text(0.01, 0.90, f"{pst}\nSurface Speed",
                transform=ax_vel.transAxes, fontsize=sizes["annot"])
    
            # -------------------------
            # Strain
            # -------------------------
            lat = Strains.lat_interp((base_y, base_x))
            ax_strain.scatter(
                base_axis[base_m], lat[base_m],
                c=lat[base_m], cmap="seismic",
                vmin=global_strain_lims[0], vmax=global_strain_lims[1],
                s=max(5, getattr(Args, "point_size", 10))
            )
            ax_strain.set_ylim(global_strain_lims[0], global_strain_lims[1])
            ax_strain.text(0.01, 0.90, "Lateral Strain", transform=ax_strain.transAxes, fontsize=18)
    
            # -------------------------
            # Melt
            # -------------------------
            B2D = Interpolators.real_bed_interp
            S2D = Interpolators.real_surf_interp
            thickness = S2D((base_x, base_y)) - B2D((base_x, base_y))
    
            theta = np.zeros_like(speed, dtype=float)
            mask = (speed > Args.low_thresh) & (speed < Args.upper_thresh)
            denom = (Args.upper_thresh - Args.low_thresh)
            theta[mask] = (speed[mask] - Args.low_thresh) / denom
            theta[speed >= Args.upper_thresh] = 1.0
    
            melt_mode = getattr(Args, "melt_mode", "deterministic")
            
            if melt_mode == "statistical":
                import stat_melt as stm
                
                print('stat melt is being run \n')
                
                palette = ["#D8DBE2", "#A9BCD0", "#58A4B0", "#373F51", "#1B1B1E"]
                LIGHT_FILL  = palette[1]   # A9BCD0 (light)
                LIGHT_EDGE  = palette[2]   # 58A4B0 (still light, good outline)
                TEXT_COLOR  = palette[0]   # D8DBE2 (very light, good on dark PPT background)
                
                # Light grid (optional, looks good on dark backgrounds)
                ax_tau.grid(True, alpha=0.18, linewidth=0.8, color=TEXT_COLOR)
                
                Args.Thickness = thickness
                Args.surf_vel = speed
                Args.theta = theta
                melt_mc_mm, _ = stm.statistical_melt(Args) # calcs suite of melts
                mean = np.nanmean(melt_mc_mm, axis=0)
                ax_tau.plot(base_axis[base_m], mean[base_m], linewidth=2.5, color= palette[3]) # plots mean
                ax_tau.set_ylim(global_melt_lims[0], global_melt_lims[1])
                
                i = 0
                while i < int(Args.N):
                    ax_tau.plot(base_axis[base_m], melt_mc_mm[i][base_m], alpha = 0.2, color = palette[2])
                    i += 10
                    
                mean = np.mean(melt_mc_mm, axis = 0)
                std = np.std(melt_mc_mm, axis=0)
                
                print(type(std))
                print(std)
                
                print('\n\n\n',mean)
                print(type(mean))
                
                ax_tau.fill_between(
                    base_axis[base_m], mean[base_m] + 2*std[base_m], mean[base_m] - 2*std[base_m],
                    color=palette[1],
                    alpha=0.35,
                    label=r"2$\sigma$"
                )
                
                Corrected_PST_list[pst]["bmelt"] = mean
                Corrected_PST_list[pst]["bmelt_all"] = melt_mc_mm
                
                # for pst in Corrected_PST_list:
                #     Corrected_PST_list[pst].pop("bmelt", None)
                #     Corrected_PST_list[pst].pop("bmelt_all", None)
                
            else:
                tau_b = Args.Bo * (
                    ((Args.m * Args.n + 1) / (2 * thickness)) * speed * (1 - theta)
                ) ** (1 / Args.n)
                s2yr = (60 ** 2) * 24 * 365.25
                G, dTdz, k, rho, L = 0.06, 0.02, 2.2, 917.0, 3.34e5
                u_b = speed * theta
                mrate = (1 / (rho * L)) * (G - k * dTdz + (1 / s2yr) * tau_b * u_b)
                melt_in_mm = mrate * s2yr * 1000
    
                ax_tau.scatter(
                    base_axis[base_m], melt_in_mm[base_m],
                    c=melt_in_mm[base_m], cmap="cmo.thermal" if "cmo" in plt.colormaps() else "inferno",
                    vmin=global_melt_lims[0], vmax=global_melt_lims[1],
                    s=max(5, getattr(Args, "point_size", 10))
                )
                ax_tau.set_ylim(global_melt_lims[0], global_melt_lims[1])
                
                Corrected_PST_list[pst]["bmelt"] = melt_in_mm
    
            ax_tau.text(0.01, 0.90, "Basal Melt", transform=ax_tau.transAxes, fontsize=18)
        
        elif dataset_option == 2: 
            
            # -------------------------
            # Radar Data
            # -------------------------
            
            spec_raw = radar_df['SPEC_FILTERED']
            foc1_pow_raw = radar_df['FOC1_POWER']
            foc2_pow_raw = radar_df['FOC2_POWER']
            rad_x = radar_df['x']
            rad_y = radar_df['y']
            
            from scipy.interpolate import LinearNDInterpolator
            
            spec_interp = LinearNDInterpolator(list(zip(rad_y, rad_x)), spec_raw)  # match your (y,x) convention
            spec_plot = spec_interp((base_y, base_x))
            
            foc1_pow_interp = LinearNDInterpolator(list(zip(rad_y, rad_x)), foc1_pow_raw)
            foc1_pow_plot = foc1_pow_interp((base_y, base_x))
            
            foc2_pow_interp = LinearNDInterpolator(list(zip(rad_y, rad_x)), foc2_pow_raw)
            foc2_pow_plot = foc2_pow_interp((base_y, base_x))
            
            # coherent_interp = radar_df['Coherent Power'][2]
            # incoherent_interp = radar_df['Incoherent Power'][2]
            
            # -------------
            # Specularity
            # -------------
            
            ax_vel.scatter(
                base_axis[base_m], spec_plot[base_m],
                c= spec_plot[base_m], cmap="viridis",
                vmin=global_spec_lims[0], vmax=global_spec_lims[1],
                s=max(5, getattr(Args, "point_size", 10)))
            ax_vel.text(0.01, 0.90, "Specularity", transform=ax_vel.transAxes, fontsize=25)
            
            ax_vel.plot(
                base_axis[base_m], spec_plot[base_m])
            
            
            # -----------------
            # FOC1 Power
            # -----------------

            ax_strain.scatter(
                base_axis[base_m], foc1_pow_plot[base_m],
                c= foc1_pow_plot[base_m], cmap="viridis",
                vmin=global_foc1_pow_lims[0], vmax=global_foc1_pow_lims[1],
                s=max(5, getattr(Args, "point_size", 10)))
            ax_strain.text(0.01, 0.90, "Foc_1 Power", transform=ax_strain.transAxes, fontsize=25)
            
            ax_strain.plot(
                base_axis[base_m], foc1_pow_plot[base_m])
            
            # ------------------
              # FOC2 Power
            # ------------------
            
            ax_tau.scatter(
                base_axis[base_m], foc2_pow_plot[base_m],
                c= foc2_pow_plot[base_m], cmap="magma",
                vmin=global_foc2_pow_lims[0], vmax=global_foc2_pow_lims[1],
                s=max(5, getattr(Args, "point_size", 10))
            )
            
            ax_tau.text(0.01, 0.90, "Foc_2 Power", transform=ax_tau.transAxes, fontsize=18)
            
            ax_tau.plot(
                base_axis[base_m], foc2_pow_plot[base_m])
            
        else:
            print("ERROR: please specificy a dataset (options are 1 (ice flow) and 2 (radar))")

        # -------------------------
        # Layers
        # -------------------------
        
        layer_items = [(k, v) for k, v in layers.items() if isinstance(v, dict)][start_index:]
        
        for lname, layer in layer_items:
            
            a = np.asarray(layer.get(axis_key, np.full_like(layer["abs_distance"], np.nan)), dtype=float)
            z = _get_layer_z(layer, Args.depth)

            m = np.isfinite(a) & np.isfinite(z) & (a >= x0) & (a <= x1)
            if np.count_nonzero(m) < 2:
                continue

            a_sub = a[m]
            z_sub = z[m]

            # Fast line plotting w/ NaN breaks via masking
            a_m, z_m = _mask_sort_line(a_sub, z_sub)

            if getattr(Args, "plot_style", "line") == "scatter":
                ax_layers.scatter(a_m, z_m, s=getattr(Args, "point_size", 10), alpha=0.9)
            else:
                ax_layers.plot(a_m, z_m, linewidth=sizes["line"], alpha=0.9)

        ax_layers.text(0.01, 0.90, "Englacial Layers", transform=ax_layers.transAxes, fontsize=18)

        # Shared formatting per PST
        for ax in (ax_vel, ax_strain, ax_tau, ax_layers):
            ax.set_xlim(x0, x1)
            ax.grid(color="white", linestyle="--", linewidth=sizes["grid"], alpha=0.5)

        # labels
        if j == 0:
            
            if dataset_option == 1:
            
                ax_vel.set_ylabel("m/yr", fontsize=sizes["label"])
                ax_strain.set_ylabel("a$^{-1}$", fontsize=sizes["label"])
                ax_tau.set_ylabel("mm/yr", fontsize=sizes["label"])
                ax_layers.set_ylabel("Elevation (m)" if Args.depth == "Elevation" else "Fractional depth",
                                     fontsize=sizes["label"])
                
            elif dataset_option == 2:
                
                ax_vel.set_ylabel("(unitless)", fontsize=sizes["label"])
                ax_strain.set_ylabel("dB", fontsize=sizes["label"])
                ax_tau.set_ylabel("dB", fontsize=sizes["label"])
                ax_layers.set_ylabel("Elevation (m)" if Args.depth == "Elevation" else "Fractional depth",
                                     fontsize=sizes["label"])
                

        ax_layers.set_xlabel("Absolute distance along PST (m)", fontsize=sizes["label"])
        
        # ticks
        for ax in (ax_vel, ax_strain, ax_tau, ax_layers):
            ax.tick_params(axis="both", labelsize=sizes["tick"], width=0.8 * sizes["line"])

        # Optional: enforce consistent “radargram visual scale” note only (not forced)
        # You could enforce same ylim across PSTs by passing Args.layer_ylim or similar.

    plt.show()
    
    return {
        "global_vel_lims": global_vel_lims,
        "global_strain_lims": global_strain_lims,
        "global_melt_lims": global_melt_lims,
    }

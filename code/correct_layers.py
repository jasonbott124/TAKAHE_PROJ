# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 13 20:33:54 2026

# @author: Jason
# """
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 20:33:54 2026

@author: Jason
"""

def produce_layer_df(folder_list, projection="EPSG:3031", coord = 'latlon'):
    """
    Parameters
    ----------
    folder_list : list of (str, str)
        Example: [("X47a", r"C:\\...\\X47a_folder"), ("Y31b", r"C:\\...\\Y31b_folder")]
    projection : str
        EPSG code, default EPSG:3031.

    Returns
    -------
    PST_list : dict
        PST_list["X47a"] -> dict of layers in that PST
        PST_list["X47a"]["some_layer_file.txt"] -> layer dict with x_array,y_array,z_array,x_position
    """
    from pyproj import Transformer
    import os
    import pandas as pd
    import numpy as np

    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)

    PST_list = {}

    for pst_name, folder in folder_list.items():
        
        layers = {}  # holds all layers for THIS PST

        for file in os.listdir(folder):
            if not file.lower().endswith(".txt"):
                continue

            filepath = os.path.join(folder, file)

            df = pd.read_csv(filepath, header=None)
            
            if coord == 'latlon':
                df.columns = ["label", "col1", "col2", "lon", "lat", "elev"]
                
                east, north = transformer.transform(df["lon"].values, df["lat"].values)
                x_array = np.asarray(east, dtype=float)
                y_array = np.asarray(north, dtype=float)
                
            elif coord == 'polar-stereo':
                df.columns = ["label", "col1", "col2", "eastings", "northings", "elev"]
                x_array = np.asarray(df["eastings"], dtype=float)
                y_array = np.asarray(df["northings"], dtype=float)
            else:
                print('need to specify proper coordinate system')
                
            z_array = np.asarray(df["elev"].values, dtype=float)

            dx = np.diff(x_array)
            dy = np.diff(y_array)
            segment_lengths = np.sqrt(dx**2 + dy**2)
            x_position = np.concatenate([[0.0], np.cumsum(segment_lengths)])

            layers[file] = {
                "x_array": x_array,
                "y_array": y_array,
                "z_array": z_array,
                "x_position": x_position,
            }

        PST_list[pst_name] = layers

    return PST_list


def correct_layers_to_absolute_elevation(
    PST_list,
    surf_interp,
    bed_interp,
    *,
    z_key_in="z_array",
    z_key_out="z_array",
    thickness_min=1.0,
    keep_original=True
):
    """
    Returns all_layers_corrected where each layer gets an absolute-elevation array computed
    using Bedmap-style interpolators for surface and bed elevation.

    This mirrors the logic used in your script:
      1) Evaluate Bedmap surface S(x,y) and bed B(x,y) at each layer point.
      2) Convert the layer's *relative* vertical coordinate into a fractional depth.
      3) Convert fractional depth back to *absolute elevation* using Bedmap thickness.

    Assumptions / conventions (match your earlier workflow):
    - layer[z_key_in] is a RELATIVE vertical coordinate measured within ice thickness,
      where you previously used: z_arr = layer["z_array"] / rel_thk
      (i.e., z_key_in is like "depth below surface" or "distance from bed" expressed in meters).
    - If your z_array is already *absolute elevation* (ASL), then you should NOT use this
      converter; or you should change `mode` logic (see notes below).

    Guarantees:
    - Array sizes and indexing are preserved (no resampling).
    - Uses (x_array, y_array) at each index directly.

    Parameters
    ----------
    all_layers : dict
        {layer_name: {"x_array":..., "y_array":..., z_key_in:... , ...}, ...}

    surf_interp, bed_interp : callable
        Interpolators like RegularGridInterpolator that accept (x,y) tuples:
            S = surf_interp((x, y))
            B = bed_interp((x, y))

    z_key_in : str
        Input key holding the layer vertical coordinate to convert.

    z_key_out : str
        Output key for absolute elevation array.

    thickness_min : float
        Minimum allowed thickness (S-B) to avoid division blowups.

    keep_original : bool
        If True, returns a deepcopy of layers and appends results.
        If False, modifies a shallow copy in-place and returns it.

    Returns
    -------
    all_layers_corrected : dict
        Same structure as all_layers, but each layer has:
          - "surf_bm" : Bedmap surface sampled at layer points
          - "bed_bm"  : Bedmap bed sampled at layer points
          - "thk_bm"  : thickness = surf_bm - bed_bm
          - z_key_out : absolute elevation (ASL) at layer points
          - "frac_depth" : fractional depth used (dimensionless)
    """
    
    import copy
    import numpy as np
    
    # Copy once, outside loops
    PST_src = copy.deepcopy(PST_list) if keep_original else PST_list

    Corrected_PST_list = {}

    for PSTname, layers in PST_src.items():
    
        for lname, layer in layers.items():
            
            if z_key_in not in layer:
                continue
            
            x = np.asarray(layer["x_array"], dtype=float)
            y = np.asarray(layer["y_array"], dtype=float)
    
            if z_key_in not in layer:
                # skip quietly (or raise if you prefer)
                continue
    
            z_in = np.asarray(layer[z_key_in], dtype=float)
    
            # Sample Bedmap surface/bed at each point (same shape as x/y)
            S = np.asarray(surf_interp((x, y)), dtype=float)
            B = np.asarray(bed_interp((x, y)), dtype=float)
            thk = S - B
    
            # Validity mask (preserve indexing; output NaNs where invalid)
            ok = (
                np.isfinite(x) & np.isfinite(y) & np.isfinite(z_in) &
                np.isfinite(S) & np.isfinite(B) & np.isfinite(thk) &
                (thk >= thickness_min)
            )
    
            frac = np.full_like(z_in, np.nan, dtype=float)
            z_abs = np.full_like(z_in, np.nan, dtype=float)
            
            
    
            # --- Core conversion (matches your "z_arr = z / thk") ---
            # frac depth measured from bed (0=bed, 1=surface) if z_in = height above bed
            # absolute elevation: B + frac*thk
            #
            # If instead your z_in is depth below surface, use:
            #   frac_down = z_in / thk
            #   z_abs = S - frac_down*thk
            #
            # Your later plotting snippet uses z_arr = layer["z_array"]/rel_thk,
            # so we treat z_in as "meters within thickness from bed upward".
            frac[ok] = z_in[ok] / thk[ok]
            z_abs[ok] = S[ok] - frac[ok] * thk[ok]
            
            # z_real = real_surf - frac_down * real_thk
    
            # Store diagnostics + outputs
            layer["surf_bm"] = S
            layer["bed_bm"] = B
            layer["thk_bm"] = thk
            layer["frac_depth"] = frac
            layer[z_key_out] = z_abs
            
        Corrected_PST_list[PSTname] = layers
        
    return Corrected_PST_list


from dataclasses import dataclass

@dataclass
class PSTFiles:
    surf_file: str
    bed_file: str
    base_file: str  # default = surf_file


def auto_pick_surface_bed_and_base(
    layers_dict: dict,
    *,
    base_choice: str = "Surface",   # "surface" or "bed" or exact filename
    case_insensitive: bool = True
) -> PSTFiles:
    """
    layers_dict: one PST's layer dict, e.g. Corrected_PST_list["Y31b"]
                keys are filenames like:
                  "Bed_THW_jbott_SJB2_Y31b.txt"
                  "Surface_THW_jbott_SJB2_Y31b.txt"
                  "0,636ka_THW_SJB2_Y31b.txt"  (etc)

    base_choice:
      - "surface" (default): base_file = surface file
      - "bed": base_file = bed file
      - exact filename string: base_file = that filename (must exist)
    """

    keys = list(layers_dict.keys())

    def norm(s: str) -> str:
        return s.lower() if case_insensitive else s

    # Find candidates by standard prefixes
    surf_candidates = [k for k in keys if norm(k).startswith("surface_")]
    bed_candidates  = [k for k in keys if norm(k).startswith("bed_")]

    if len(surf_candidates) == 0:
        raise ValueError("Could not find a surface file (expected filename starting with 'Surface_').")
    if len(bed_candidates) == 0:
        raise ValueError("Could not find a bed file (expected filename starting with 'Bed_').")

    # If multiples exist, pick the one that best matches the PST suffix pattern by "longest name"
    # (usually the most specific). You can swap this strategy if needed.
    surf_file = max(surf_candidates, key=len)
    bed_file  = max(bed_candidates,  key=len)

    # Decide base
    if base_choice in ("surface", "Surface", "SURFACE"):
        base_file = surf_file
    elif base_choice in ("bed", "Bed", "BED"):
        base_file = bed_file
    else:
        # user passed a filename explicitly
        if base_choice not in layers_dict:
            raise ValueError(f"base_choice '{base_choice}' not found in this PST layers.")
        base_file = base_choice

    return PSTFiles(surf_file=surf_file, bed_file=bed_file, base_file=base_file)


def layer_absdist_from_base(
    PST_list,
    FilesByPST,
    *,
    absdist_key="abs_distance",
    base_key=None,              # optional override; otherwise uses FilesByPST[PSTname].base_file
    leafsize=40,
):
    """
    Adds an absolute-distance coordinate to every layer in every PST, stored as layer[absdist_key].

    Definition:
      - Build cumulative distance along the base layer using its (x_array, y_array).
      - For each point in each layer, assign abs_distance = abs_distance_of_nearest_base_point.

    Notes:
      - Preserves array sizes and indexing (NaNs remain NaNs).
      - Uses KDTree to avoid O(N^2) memory blowups.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    for PSTname, PST in PST_list.items():

        # pick base layer
        base_file = base_key if base_key is not None else FilesByPST[PSTname].base_file
        base = PST[base_file]

        x_base = np.asarray(base["x_array"], dtype=float)
        y_base = np.asarray(base["y_array"], dtype=float)

        okb = np.isfinite(x_base) & np.isfinite(y_base)
        idxb = np.where(okb)[0]
        if idxb.size < 2:
            print(f"[WARN] {PSTname}: base layer '{base_file}' has <2 valid points. Skipping PST.")
            continue

        # --- cumulative distance along base, preserving indexing ---
        absdist_base = np.full_like(x_base, np.nan, dtype=float)
        absdist_base[idxb[0]] = 0.0

        # vectorized over valid indices (fast)
        dx = np.diff(x_base[idxb])
        dy = np.diff(y_base[idxb])
        seg = np.hypot(dx, dy)
        absdist_base[idxb[1:]] = np.cumsum(seg)

        # build KDTree on valid base points
        xb = x_base[idxb]
        yb = y_base[idxb]
        db = absdist_base[idxb]  # length = nb_valid
        tree = cKDTree(np.column_stack([xb, yb]), leafsize=leafsize)

        # --- assign abs distance to each layer by nearest base point ---
        for lname, layer in PST.items():
            x = np.asarray(layer["x_array"], dtype=float)
            y = np.asarray(layer["y_array"], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)

            out = np.full_like(x, np.nan, dtype=float)
            if np.count_nonzero(ok) == 0:
                layer[absdist_key] = out
                continue

            q = np.column_stack([x[ok], y[ok]])
            _, nn = tree.query(q, k=1)       # nn are indices into xb/yb/db
            out[ok] = db[nn]

            layer[absdist_key] = out

    return PST_list

# Default analysis windows (abs_distance, m). Lines listed here are not prompted for.
# Edit here, or pass preset_limits={...} to override; pass preset_limits={} to be prompted for every line.
DEFAULT_LIMITS = {
    "X43a": (70000.0, 100000.0),
    "X45a": (150000.0, 180000.0),
    "X47a": (170000.0, 200000.0),
    "X49b": (70000.0, 100000.0),
    "Y15b": (95000.0, 115000.0),
    "Y17a": (107000.0, 137000.0),
    "Y19a": (265000.0, 295000.0),
    "Y21a": (200000.0, 230000.0),
    "Y21b": (0.0, 30000.0),
    "Y23a": (260000.0, 290000.0),
}


def pick_limits_by_pst(
    Corrected_PST_list,
    FilesByPST,
    ArgumentsByPST=None,
    *,
    preset_limits=None,           # {PST: (xmin, xmax)}; None = DEFAULT_LIMITS; {} = prompt for all
    show_presets=False,           # True = also plot each preset window (no prompt)
    skip_existing=False,          # True = don't re-prompt a PST that already has limits_abs stored
    axis="abs_distance",          # expects layer[axis] already exists
    elev_type="true_elevation", # "fractional_depth" or "true_elevation"
    start_index=0,
    plot_style="line",            # "line" or "scatter"
    sample_frac=0.30,             # only used for plotting
    seed=0,
    figsize=(16, 5),
    line_width=1.25,
    alpha=0.9,
):
    """
    Interactive PST-by-PST limit picker.

    Workflow per PST:
      1) Plot layers vs abs_distance (using existing arrays; no abs_distance computation here).
      2) Prompt user for x-limits (xmin xmax).
      3) Replot ONLY data within limits.
      4) Ask: "Are you satisfied with these limits? (Y/N)"
         - If Y: store to ArgumentsByPST[PSTname].limits_abs = (xmin, xmax) and move on.
         - If N: repeat step 2.

    Preset limits:
      - PSTs found in preset_limits (default: DEFAULT_LIMITS above) are stored directly,
        with no plot or prompt. Any PST NOT in the presets (e.g. Y67a) falls back to the
        interactive prompt as before. A malformed preset also falls back to the prompt.

    Notes / best practice:
      - For cross-PST comparisons, keeping a consistent scale is *recommended* (esp. layer panel),
        but not strictly required.

    Returns
    -------
    ArgumentsByPST : dict
        Dict of PSTname -> Arguments object with a .limits_abs attribute (tuple).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------
    # Small local "Arguments" container
    # -------------------------
    class _Args:
        def __init__(self):
            self.limits_abs = None

    if ArgumentsByPST is None:
        ArgumentsByPST = {}

    rng = np.random.default_rng(seed)

    # -------------------------
    # Helpers
    # -------------------------
    def _get_z(layer):
        if elev_type == "fractional_depth":
            # allow either key name
            if "depth_frac" in layer:
                return np.asarray(layer["depth_frac"], dtype=float)
            return np.asarray(layer.get("frac_depth", np.full_like(layer[axis], np.nan)), dtype=float)
        elif elev_type == "true_elevation":
            return np.asarray(layer.get("z_abs", layer.get("z_array", np.full_like(layer[axis], np.nan))), dtype=float)
        else:
            raise ValueError("elev_type must be 'fractional_depth' or 'true_elevation'")

    def _masked_plot(ax, x, z, *, label=None):
        # break lines at NaNs without scatter
        xm = np.ma.masked_invalid(x)
        zm = np.ma.masked_invalid(z)
        order = np.argsort(xm)
        xm = xm[order]
        zm = zm[order]

        if plot_style == "scatter":
            ax.scatter(xm, zm, s=10, alpha=alpha, label=label)
        else:
            ax.plot(xm, zm, linewidth=line_width, alpha=alpha, label=label)

    def _plot_pst(PSTname, PST, x_limits=None, title_suffix=""):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # plot all layers (sample for speed)
        items = list(PST.items())[start_index:]
        for lname, layer in items:
            if axis not in layer:
                continue

            x_full = np.asarray(layer[axis], dtype=float)
            z_full = _get_z(layer)

            ok = np.isfinite(x_full) & np.isfinite(z_full)
            if x_limits is not None:
                ok = ok & (x_full >= x_limits[0]) & (x_full <= x_limits[1])

            idx = np.where(ok)[0]
            if idx.size < 2:
                continue

            # random sample indices so x/z stay consistent
            ns = max(2, int(np.ceil(sample_frac * idx.size)))
            if ns < idx.size:
                idx_s = rng.choice(idx, size=ns, replace=False)
            else:
                idx_s = idx

            x = x_full[idx_s]
            z = z_full[idx_s]

            _masked_plot(ax, x, z, label=None)

        # cosmetics
        ax.set_xlabel("Absolute distance along radar line (m)")
        ax.set_ylabel("Fractional depth" if elev_type == "fractional_depth" else "Elevation (m)")
        ax.grid(True, alpha=0.25)

        if x_limits is None:
            ax.set_title(f"{PSTname} — pick limits{title_suffix}")
        else:
            ax.set_title(f"{PSTname} — limits = [{x_limits[0]:.1f}, {x_limits[1]:.1f}]{title_suffix}")

        plt.tight_layout()
        plt.show()

    def _parse_limits(s):
        parts = s.replace(",", " ").split()
        if len(parts) != 2:
            return None
        try:
            a, b = float(parts[0]), float(parts[1])
        except Exception:
            return None
        return (min(a, b), max(a, b))

    # -------------------------
    # Main loop
    # -------------------------
    print("\n[NOTE] Best practice: keep a consistent x/y scale across PSTs for comparing radar-layer geometry.")
    print("       Not strictly required, but it makes comparisons much easier.\n")

    presets = DEFAULT_LIMITS if preset_limits is None else preset_limits

    for PSTname, PST in Corrected_PST_list.items():

        # Ensure Arguments object exists
        if PSTname not in ArgumentsByPST:
            ArgumentsByPST[PSTname] = _Args()

        # 1) preset limits -> store and move on
        if presets.get(PSTname) is not None:
            try:
                a, b = (float(v) for v in presets[PSTname])
                x_limits = (min(a, b), max(a, b))
            except Exception:
                print(f"[WARN] {PSTname}: could not read preset {presets[PSTname]!r}; prompting instead.")
            else:
                ArgumentsByPST[PSTname].limits_abs = x_limits
                print(f"[preset] {PSTname}.limits_abs = {x_limits}")
                if show_presets:
                    _plot_pst(PSTname, PST, x_limits=x_limits, title_suffix=" (preset)")
                continue

        # 2) already picked earlier in this session -> optionally keep it
        if skip_existing and getattr(ArgumentsByPST[PSTname], "limits_abs", None) is not None:
            print(f"[keep] {PSTname}.limits_abs = {ArgumentsByPST[PSTname].limits_abs}")
            continue

        # 3) no preset -> interactive prompt, as before

        # First plot (full-range)
        _plot_pst(PSTname, PST, x_limits=None)

        # Determine a suggested range using base_file axis
        base_file = FilesByPST[PSTname].base_file
        base_layer = PST.get(base_file, None)
        if base_layer is not None and axis in base_layer:
            a0 = np.asarray(base_layer[axis], dtype=float)
            a_min = np.nanmin(a0)
            a_max = np.nanmax(a0)
            print(f"{PSTname}: suggested {axis} range from base_file ({base_file}) = [{a_min:.1f}, {a_max:.1f}]")
        else:
            print(f"{PSTname}: (could not infer suggested range from base_file)")

        # Interactive accept/reject loop
        while True:
            raw = input(f"\n{PSTname} → Enter limits (xmin xmax) in meters: ").strip()
            x_limits = _parse_limits(raw)
            if x_limits is None:
                print("  [ERROR] Could not parse. Example: 50000 120000")
                continue

            # Replot within limits
            _plot_pst(PSTname, PST, x_limits=x_limits)

            sat = input(f"{PSTname} → Are you satisfied with these limits? (Y/N): ").strip().lower()
            if sat in ("y", "yes"):
                ArgumentsByPST[PSTname].limits_abs = x_limits
                print(f"  [OK] Stored {PSTname}.limits_abs = {x_limits}")
                break
            else:
                print("  [INFO] Noted — enter new limits.")

    return ArgumentsByPST



# -----------------------------
# Example: build FilesByPST dict
# -----------------------------
# FilesByPST = {}
# for pst_name, layers in Corrected_PST_list.items():
#     FilesByPST[pst_name] = auto_pick_surface_bed_and_base(layers, base_choice="surface")
#
# # User can override base_file later:
# FilesByPST["Y31b"].base_file = "0,636ka_THW_SJB2_Y31b.txt"   # example override
# def produce_layer_df(folder_list, projection="EPSG:3031", coord = 'latlon'):
#     """
#     Parameters
#     ----------
#     folder_list : list of (str, str)
#         Example: [("X47a", r"C:\\...\\X47a_folder"), ("Y31b", r"C:\\...\\Y31b_folder")]
#     projection : str
#         EPSG code, default EPSG:3031.

#     Returns
#     -------
#     PST_list : dict
#         PST_list["X47a"] -> dict of layers in that PST
#         PST_list["X47a"]["some_layer_file.txt"] -> layer dict with x_array,y_array,z_array,x_position
#     """
#     from pyproj import Transformer
#     import os
#     import pandas as pd
#     import numpy as np

#     transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)

#     PST_list = {}

#     for pst_name, folder in folder_list.items():
        
#         layers = {}  # holds all layers for THIS PST

#         for file in os.listdir(folder):
#             if not file.lower().endswith(".txt"):
#                 continue

#             filepath = os.path.join(folder, file)

#             df = pd.read_csv(filepath, header=None)
            
#             if coord == 'latlon':
#                 df.columns = ["label", "col1", "col2", "lon", "lat", "elev"]
                
#                 east, north = transformer.transform(df["lon"].values, df["lat"].values)
#                 x_array = np.asarray(east, dtype=float)
#                 y_array = np.asarray(north, dtype=float)
                
#             elif coord == 'polar-stereo':
#                 df.columns = ["label", "col1", "col2", "eastings", "northings", "elev"]
#                 x_array = np.asarray(df["eastings"], dtype=float)
#                 y_array = np.asarray(df["northings"], dtype=float)
#             else:
#                 print('need to specify proper coordinate system')
                
#             z_array = np.asarray(df["elev"].values, dtype=float)

#             dx = np.diff(x_array)
#             dy = np.diff(y_array)
#             segment_lengths = np.sqrt(dx**2 + dy**2)
#             x_position = np.concatenate([[0.0], np.cumsum(segment_lengths)])

#             layers[file] = {
#                 "x_array": x_array,
#                 "y_array": y_array,
#                 "z_array": z_array,
#                 "x_position": x_position,
#             }

#         PST_list[pst_name] = layers

#     return PST_list


# def correct_layers_to_absolute_elevation(
#     PST_list,
#     surf_interp,
#     bed_interp,
#     *,
#     z_key_in="z_array",
#     z_key_out="z_array",
#     thickness_min=1.0,
#     keep_original=True
# ):
#     """
#     Returns all_layers_corrected where each layer gets an absolute-elevation array computed
#     using Bedmap-style interpolators for surface and bed elevation.

#     This mirrors the logic used in your script:
#       1) Evaluate Bedmap surface S(x,y) and bed B(x,y) at each layer point.
#       2) Convert the layer's *relative* vertical coordinate into a fractional depth.
#       3) Convert fractional depth back to *absolute elevation* using Bedmap thickness.

#     Assumptions / conventions (match your earlier workflow):
#     - layer[z_key_in] is a RELATIVE vertical coordinate measured within ice thickness,
#       where you previously used: z_arr = layer["z_array"] / rel_thk
#       (i.e., z_key_in is like "depth below surface" or "distance from bed" expressed in meters).
#     - If your z_array is already *absolute elevation* (ASL), then you should NOT use this
#       converter; or you should change `mode` logic (see notes below).

#     Guarantees:
#     - Array sizes and indexing are preserved (no resampling).
#     - Uses (x_array, y_array) at each index directly.

#     Parameters
#     ----------
#     all_layers : dict
#         {layer_name: {"x_array":..., "y_array":..., z_key_in:... , ...}, ...}

#     surf_interp, bed_interp : callable
#         Interpolators like RegularGridInterpolator that accept (x,y) tuples:
#             S = surf_interp((x, y))
#             B = bed_interp((x, y))

#     z_key_in : str
#         Input key holding the layer vertical coordinate to convert.

#     z_key_out : str
#         Output key for absolute elevation array.

#     thickness_min : float
#         Minimum allowed thickness (S-B) to avoid division blowups.

#     keep_original : bool
#         If True, returns a deepcopy of layers and appends results.
#         If False, modifies a shallow copy in-place and returns it.

#     Returns
#     -------
#     all_layers_corrected : dict
#         Same structure as all_layers, but each layer has:
#           - "surf_bm" : Bedmap surface sampled at layer points
#           - "bed_bm"  : Bedmap bed sampled at layer points
#           - "thk_bm"  : thickness = surf_bm - bed_bm
#           - z_key_out : absolute elevation (ASL) at layer points
#           - "frac_depth" : fractional depth used (dimensionless)
#     """
    
#     import copy
#     import numpy as np
    
#     # Copy once, outside loops
#     PST_src = copy.deepcopy(PST_list) if keep_original else PST_list

#     Corrected_PST_list = {}

#     for PSTname, layers in PST_src.items():
    
#         for lname, layer in layers.items():
            
#             if z_key_in not in layer:
#                 continue
            
#             x = np.asarray(layer["x_array"], dtype=float)
#             y = np.asarray(layer["y_array"], dtype=float)
    
#             if z_key_in not in layer:
#                 # skip quietly (or raise if you prefer)
#                 continue
    
#             z_in = np.asarray(layer[z_key_in], dtype=float)
    
#             # Sample Bedmap surface/bed at each point (same shape as x/y)
#             S = np.asarray(surf_interp((x, y)), dtype=float)
#             B = np.asarray(bed_interp((x, y)), dtype=float)
#             thk = S - B
    
#             # Validity mask (preserve indexing; output NaNs where invalid)
#             ok = (
#                 np.isfinite(x) & np.isfinite(y) & np.isfinite(z_in) &
#                 np.isfinite(S) & np.isfinite(B) & np.isfinite(thk) &
#                 (thk >= thickness_min)
#             )
    
#             frac = np.full_like(z_in, np.nan, dtype=float)
#             z_abs = np.full_like(z_in, np.nan, dtype=float)
            
            
    
#             # --- Core conversion (matches your "z_arr = z / thk") ---
#             # frac depth measured from bed (0=bed, 1=surface) if z_in = height above bed
#             # absolute elevation: B + frac*thk
#             #
#             # If instead your z_in is depth below surface, use:
#             #   frac_down = z_in / thk
#             #   z_abs = S - frac_down*thk
#             #
#             # Your later plotting snippet uses z_arr = layer["z_array"]/rel_thk,
#             # so we treat z_in as "meters within thickness from bed upward".
#             frac[ok] = z_in[ok] / thk[ok]
#             z_abs[ok] = S[ok] - frac[ok] * thk[ok]
            
#             # z_real = real_surf - frac_down * real_thk
    
#             # Store diagnostics + outputs
#             layer["surf_bm"] = S
#             layer["bed_bm"] = B
#             layer["thk_bm"] = thk
#             layer["frac_depth"] = frac
#             layer[z_key_out] = z_abs
            
#         Corrected_PST_list[PSTname] = layers
        
#     return Corrected_PST_list


# from dataclasses import dataclass

# @dataclass
# class PSTFiles:
#     surf_file: str
#     bed_file: str
#     base_file: str  # default = surf_file


# def auto_pick_surface_bed_and_base(
#     layers_dict: dict,
#     *,
#     base_choice: str = "Surface",   # "surface" or "bed" or exact filename
#     case_insensitive: bool = True
# ) -> PSTFiles:
#     """
#     layers_dict: one PST's layer dict, e.g. Corrected_PST_list["Y31b"]
#                 keys are filenames like:
#                   "Bed_THW_jbott_SJB2_Y31b.txt"
#                   "Surface_THW_jbott_SJB2_Y31b.txt"
#                   "0,636ka_THW_SJB2_Y31b.txt"  (etc)

#     base_choice:
#       - "surface" (default): base_file = surface file
#       - "bed": base_file = bed file
#       - exact filename string: base_file = that filename (must exist)
#     """

#     keys = list(layers_dict.keys())

#     def norm(s: str) -> str:
#         return s.lower() if case_insensitive else s

#     # Find candidates by standard prefixes
#     surf_candidates = [k for k in keys if norm(k).startswith("surface_")]
#     bed_candidates  = [k for k in keys if norm(k).startswith("bed_")]

#     if len(surf_candidates) == 0:
#         raise ValueError("Could not find a surface file (expected filename starting with 'Surface_').")
#     if len(bed_candidates) == 0:
#         raise ValueError("Could not find a bed file (expected filename starting with 'Bed_').")

#     # If multiples exist, pick the one that best matches the PST suffix pattern by "longest name"
#     # (usually the most specific). You can swap this strategy if needed.
#     surf_file = max(surf_candidates, key=len)
#     bed_file  = max(bed_candidates,  key=len)

#     # Decide base
#     if base_choice in ("surface", "Surface", "SURFACE"):
#         base_file = surf_file
#     elif base_choice in ("bed", "Bed", "BED"):
#         base_file = bed_file
#     else:
#         # user passed a filename explicitly
#         if base_choice not in layers_dict:
#             raise ValueError(f"base_choice '{base_choice}' not found in this PST layers.")
#         base_file = base_choice

#     return PSTFiles(surf_file=surf_file, bed_file=bed_file, base_file=base_file)


# def layer_absdist_from_base(
#     PST_list,
#     FilesByPST,
#     *,
#     absdist_key="abs_distance",
#     base_key=None,              # optional override; otherwise uses FilesByPST[PSTname].base_file
#     leafsize=40,
# ):
#     """
#     Adds an absolute-distance coordinate to every layer in every PST, stored as layer[absdist_key].

#     Definition:
#       - Build cumulative distance along the base layer using its (x_array, y_array).
#       - For each point in each layer, assign abs_distance = abs_distance_of_nearest_base_point.

#     Notes:
#       - Preserves array sizes and indexing (NaNs remain NaNs).
#       - Uses KDTree to avoid O(N^2) memory blowups.
#     """
#     import numpy as np
#     from scipy.spatial import cKDTree

#     for PSTname, PST in PST_list.items():

#         # pick base layer
#         base_file = base_key if base_key is not None else FilesByPST[PSTname].base_file
#         base = PST[base_file]

#         x_base = np.asarray(base["x_array"], dtype=float)
#         y_base = np.asarray(base["y_array"], dtype=float)

#         okb = np.isfinite(x_base) & np.isfinite(y_base)
#         idxb = np.where(okb)[0]
#         if idxb.size < 2:
#             print(f"[WARN] {PSTname}: base layer '{base_file}' has <2 valid points. Skipping PST.")
#             continue

#         # --- cumulative distance along base, preserving indexing ---
#         absdist_base = np.full_like(x_base, np.nan, dtype=float)
#         absdist_base[idxb[0]] = 0.0

#         # vectorized over valid indices (fast)
#         dx = np.diff(x_base[idxb])
#         dy = np.diff(y_base[idxb])
#         seg = np.hypot(dx, dy)
#         absdist_base[idxb[1:]] = np.cumsum(seg)

#         # build KDTree on valid base points
#         xb = x_base[idxb]
#         yb = y_base[idxb]
#         db = absdist_base[idxb]  # length = nb_valid
#         tree = cKDTree(np.column_stack([xb, yb]), leafsize=leafsize)

#         # --- assign abs distance to each layer by nearest base point ---
#         for lname, layer in PST.items():
#             x = np.asarray(layer["x_array"], dtype=float)
#             y = np.asarray(layer["y_array"], dtype=float)
#             ok = np.isfinite(x) & np.isfinite(y)

#             out = np.full_like(x, np.nan, dtype=float)
#             if np.count_nonzero(ok) == 0:
#                 layer[absdist_key] = out
#                 continue

#             q = np.column_stack([x[ok], y[ok]])
#             _, nn = tree.query(q, k=1)       # nn are indices into xb/yb/db
#             out[ok] = db[nn]

#             layer[absdist_key] = out

#     return PST_list

# def pick_limits_by_pst(
#     Corrected_PST_list,
#     FilesByPST,
#     ArgumentsByPST=None,
#     *,
#     axis="abs_distance",          # expects layer[axis] already exists
#     elev_type="true_elevation", # "fractional_depth" or "true_elevation"
#     start_index=0,
#     plot_style="line",            # "line" or "scatter"
#     sample_frac=0.30,             # only used for plotting
#     seed=0,
#     figsize=(16, 5),
#     line_width=1.25,
#     alpha=0.9,
# ):
#     """
#     Interactive PST-by-PST limit picker.

#     Workflow per PST:
#       1) Plot layers vs abs_distance (using existing arrays; no abs_distance computation here).
#       2) Prompt user for x-limits (xmin xmax).
#       3) Replot ONLY data within limits.
#       4) Ask: "Are you satisfied with these limits? (Y/N)"
#          - If Y: store to ArgumentsByPST[PSTname].limits_abs = (xmin, xmax) and move on.
#          - If N: repeat step 2.

#     Notes / best practice:
#       - For cross-PST comparisons, keeping a consistent scale is *recommended* (esp. layer panel),
#         but not strictly required.

#     Returns
#     -------
#     ArgumentsByPST : dict
#         Dict of PSTname -> Arguments object with a .limits_abs attribute (tuple).
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # -------------------------
#     # Small local "Arguments" container
#     # -------------------------
#     class _Args:
#         def __init__(self):
#             self.limits_abs = None

#     if ArgumentsByPST is None:
#         ArgumentsByPST = {}

#     rng = np.random.default_rng(seed)

#     # -------------------------
#     # Helpers
#     # -------------------------
#     def _get_z(layer):
#         if elev_type == "fractional_depth":
#             # allow either key name
#             if "depth_frac" in layer:
#                 return np.asarray(layer["depth_frac"], dtype=float)
#             return np.asarray(layer.get("frac_depth", np.full_like(layer[axis], np.nan)), dtype=float)
#         elif elev_type == "true_elevation":
#             return np.asarray(layer.get("z_abs", layer.get("z_array", np.full_like(layer[axis], np.nan))), dtype=float)
#         else:
#             raise ValueError("elev_type must be 'fractional_depth' or 'true_elevation'")

#     def _masked_plot(ax, x, z, *, label=None):
#         # break lines at NaNs without scatter
#         xm = np.ma.masked_invalid(x)
#         zm = np.ma.masked_invalid(z)
#         order = np.argsort(xm)
#         xm = xm[order]
#         zm = zm[order]

#         if plot_style == "scatter":
#             ax.scatter(xm, zm, s=10, alpha=alpha, label=label)
#         else:
#             ax.plot(xm, zm, linewidth=line_width, alpha=alpha, label=label)

#     def _plot_pst(PSTname, PST, x_limits=None, title_suffix=""):
#         plt.style.use("dark_background")
#         fig, ax = plt.subplots(figsize=figsize)
#         fig.patch.set_alpha(0)
#         ax.set_facecolor("none")

#         # plot all layers (sample for speed)
#         items = list(PST.items())[start_index:]
#         for lname, layer in items:
#             if axis not in layer:
#                 continue

#             x_full = np.asarray(layer[axis], dtype=float)
#             z_full = _get_z(layer)

#             ok = np.isfinite(x_full) & np.isfinite(z_full)
#             if x_limits is not None:
#                 ok = ok & (x_full >= x_limits[0]) & (x_full <= x_limits[1])

#             idx = np.where(ok)[0]
#             if idx.size < 2:
#                 continue

#             # random sample indices so x/z stay consistent
#             ns = max(2, int(np.ceil(sample_frac * idx.size)))
#             if ns < idx.size:
#                 idx_s = rng.choice(idx, size=ns, replace=False)
#             else:
#                 idx_s = idx

#             x = x_full[idx_s]
#             z = z_full[idx_s]

#             _masked_plot(ax, x, z, label=None)

#         # cosmetics
#         ax.set_xlabel("Absolute distance along radar line (m)")
#         ax.set_ylabel("Fractional depth" if elev_type == "fractional_depth" else "Elevation (m)")
#         ax.grid(True, alpha=0.25)

#         if x_limits is None:
#             ax.set_title(f"{PSTname} — pick limits{title_suffix}")
#         else:
#             ax.set_title(f"{PSTname} — limits = [{x_limits[0]:.1f}, {x_limits[1]:.1f}]{title_suffix}")

#         plt.tight_layout()
#         plt.show()

#     def _parse_limits(s):
#         parts = s.replace(",", " ").split()
#         if len(parts) != 2:
#             return None
#         try:
#             a, b = float(parts[0]), float(parts[1])
#         except Exception:
#             return None
#         return (min(a, b), max(a, b))

#     # -------------------------
#     # Main loop
#     # -------------------------
#     print("\n[NOTE] Best practice: keep a consistent x/y scale across PSTs for comparing radar-layer geometry.")
#     print("       Not strictly required, but it makes comparisons much easier.\n")

#     for PSTname, PST in Corrected_PST_list.items():

#         # Ensure Arguments object exists
#         if PSTname not in ArgumentsByPST:
#             ArgumentsByPST[PSTname] = _Args()

#         # First plot (full-range)
#         _plot_pst(PSTname, PST, x_limits=None)

#         # Determine a suggested range using base_file axis
#         base_file = FilesByPST[PSTname].base_file
#         base_layer = PST.get(base_file, None)
#         if base_layer is not None and axis in base_layer:
#             a0 = np.asarray(base_layer[axis], dtype=float)
#             a_min = np.nanmin(a0)
#             a_max = np.nanmax(a0)
#             print(f"{PSTname}: suggested {axis} range from base_file ({base_file}) = [{a_min:.1f}, {a_max:.1f}]")
#         else:
#             print(f"{PSTname}: (could not infer suggested range from base_file)")

#         # Interactive accept/reject loop
#         while True:
#             raw = input(f"\n{PSTname} → Enter limits (xmin xmax) in meters: ").strip()
#             x_limits = _parse_limits(raw)
#             if x_limits is None:
#                 print("  [ERROR] Could not parse. Example: 50000 120000")
#                 continue

#             # Replot within limits
#             _plot_pst(PSTname, PST, x_limits=x_limits)

#             sat = input(f"{PSTname} → Are you satisfied with these limits? (Y/N): ").strip().lower()
#             if sat in ("y", "yes"):
#                 ArgumentsByPST[PSTname].limits_abs = x_limits
#                 print(f"  [OK] Stored {PSTname}.limits_abs = {x_limits}")
#                 break
#             else:
#                 print("  [INFO] Noted — enter new limits.")

#     return ArgumentsByPST



# # -----------------------------
# # Example: build FilesByPST dict
# # -----------------------------
# # FilesByPST = {}
# # for pst_name, layers in Corrected_PST_list.items():
# #     FilesByPST[pst_name] = auto_pick_surface_bed_and_base(layers, base_choice="surface")
# #
# # # User can override base_file later:
# # FilesByPST["Y31b"].base_file = "0,636ka_THW_SJB2_Y31b.txt"   # example override
     

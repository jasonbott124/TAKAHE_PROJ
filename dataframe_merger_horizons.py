# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 15:17:18 2026

@author: Jason
"""

# %%

def dataframe_combiner(Lists):
    
    name_list = Lists.name_list
    age_list  = Lists.age_list 
    
    ml_horizons = Lists.list_1
    ffr_dfs = Lists.list_2

    import re
    import numpy as np
    import pandas as pd
    
    def parse_age_ka(age_label: str) -> float:
        """
        Extracts the first number in the age label as ka.
        Example: 'Isochron age: 4.711 +/- 0.246 ka' -> 4.711
        """
        m = re.search(r"([-+]?\d*\.?\d+)", age_label)
        return float(m.group(1)) if m else np.nan
    
    
    # -------------------------
    # 1) Build AGE MAPS
    # -------------------------
    # You already have these:
    # name_list (Muldoon/Leuro layer names) and age_list aligned by index.
    # Example: age_list[k] corresponds to name_list[k]
    
    age_map_ml = {name_list[i]: age_list[i] for i in range(len(name_list))}
    
    # For FFR layers: YOU must decide how to map ages.
    # If FFR1..FFR6 correspond to the same ages in age_list, do this:
    ffr_layers_in_order = ["FFR1", "FFR2", "FFR3", "FFR4", "FFR5", "FFR6"]
    age_map_ffr = {ffr_layers_in_order[i]: age_list[i] for i in range(len(ffr_layers_in_order))}
    
    # If that mapping is NOT correct, just edit age_map_ffr manually, e.g.:
    # age_map_ffr = {
    #     "FFR1": "Isochron age: ... ka",
    #     "FFR2": "Isochron age: ... ka",
    #     ...
    # }
    
    
    # -------------------------
    # 2) Convert ml_horizons -> DataFrame
    # -------------------------
    ml_rows = []
    for layer, d in ml_horizons.items():
        x = np.asarray(d["x_array"])
        y = np.asarray(d["y_array"])
        z = np.asarray(d["z_array"])
    
        age_label = age_map_ml.get(layer, None)
        age_ka = parse_age_ka(age_label) if age_label is not None else np.nan
    
        df_layer = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,
            "source": "MuldoonLeuro",
            "layer": layer,
            "age_label": age_label,
            "age_ka": age_ka,
        })
        ml_rows.append(df_layer)
    
    ml_df = pd.concat(ml_rows, ignore_index=True)
    
    
    # -------------------------
    # 3) Convert ffr_dfs -> DataFrame
    # -------------------------
    ffr_rows = []
    for layer, df in ffr_dfs.items():
        # Expecting columns ['x','y','z'] (adjust here if yours differ)
        df_layer = df.copy()
    
        # Standardize columns (in case capitalization differs)
        df_layer = df_layer.rename(columns={"X":"x","Y":"y","Z":"z"})
    
        age_label = age_map_ffr.get(layer, None)
        age_ka = parse_age_ka(age_label) if age_label is not None else np.nan
    
        df_layer["source"] = "FFR"
        df_layer["layer"] = layer
        df_layer["age_label"] = age_label
        df_layer["age_ka"] = age_ka
    
        ffr_rows.append(df_layer[["x","y","z","source","layer","age_label","age_ka"]])
    
    ffr_df = pd.concat(ffr_rows, ignore_index=True)
    
    
    # -------------------------
    # 4) Merge + sort by age
    # -------------------------
    merged_layers_df = pd.concat([ml_df, ffr_df], ignore_index=True)
    
    # Clean bad z if you want (same rule you used)
    merged_layers_df.loc[merged_layers_df["z"] < -10000, "z"] = np.nan
    
    # Primary sort: age, then source, then layer (stable sorting)
    merged_layers_df = merged_layers_df.sort_values(
        by=["age_ka", "source", "layer"],
        kind="mergesort"  # stable
    ).reset_index(drop=True)
    
    # merged_layers_df is your merged, age-ordered master table
    
    custom_layer_order = Lists.custom_order
    
    # Anything not listed will fall to the bottom (still sorted by age_ka afterward if you want)
    merged_layers_df["layer_order"] = pd.Categorical(
        merged_layers_df["layer"],
        categories=custom_layer_order,
        ordered=True
    )
    
    merged_custom = merged_layers_df.sort_values(
        by=["layer_order", "age_ka", "source"],
        kind="mergesort"
    ).reset_index(drop=True)
    
    # merged_custom is now in your custom order

    
    return merged_layers_df

import config  # central data/output paths (config.py)
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:22:08 2025

@author: Jason
"""

def horizon_scraper(file_path):
    import pandas as pd
    from io import StringIO
    from pyproj import Proj, transform

    def clean_headers(df):
        return df[~(df == df.columns).all(axis=1)]

    # Header row indices (0-based)
    row_indices = {
        'FFR1': 217432,
        'FFR2': 486250,
        'FFR3': 2208,
        'FFR4': 15939,
        'FFR5': 23030,
        'FFR6': 115958,
        'FFR_surface': 1004695,
    }

    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Sort by row index
    sorted_keys = sorted(row_indices, key=lambda k: row_indices[k])
    starts = [row_indices[k] for k in sorted_keys]
    ends = starts[1:] + [len(lines)]  # next start or EOF

    horizons = {}

    for name, start, end in zip(sorted_keys, starts, ends):
        block = lines[start:end]
        df = pd.read_csv(StringIO("".join(block)))
        df = clean_headers(df)
        df = df.iloc[:, [3, 4, 5]]  # Keep only columns D, E, F
        df.columns = ['x', 'y', 'z']
        # Convert all values to numeric, coercing strings like 'NEW' to NaN
        df_cleaned = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop any rows with NaNs (i.e., originally non-numeric values)
        df_cleaned = df_cleaned.dropna()
        
        horizons[name] = df_cleaned
        

    return horizons


dfs = horizon_scraper(config.p(r"FFR_Horizons_WITH_SURFACE.csv"))

string_locs = dfs['FFR1'].applymap(lambda x: isinstance(x, str) and x == 'NEW').stack()
string_locations = string_locs[string_locs].index.tolist()

print("String locations:", string_locations)

dfs['FFR1'].head(2450)
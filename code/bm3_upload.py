# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:57:36 2026

@author: Jason
"""
import config  # central data/output paths (config.py)
import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
import Flowband_Functions as ff
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

# SCRIPT FOR UPLOADING BM3

if "bm3" in globals():
    del bm3
    import gc
    gc.collect()

def load_bedmap3(bm3_bed_path, bm3_surf_path, sub_idxs):
    
    surf = ff.new_tif_to_array(bm3_surf_path)
    bed  = ff.new_tif_to_array(bm3_bed_path)

    surf_sub = surf[sub_idxs[0]:sub_idxs[1], sub_idxs[2]:sub_idxs[3]].astype(float)
    surf_sub[surf_sub < -5000] = np.nan

    bed_sub = bed[sub_idxs[0]:sub_idxs[1], sub_idxs[2]:sub_idxs[3]].astype(float)
    bed_sub[bed_sub < -5000] = np.nan

    xc = np.arange(-3333500, 3333500, 500)
    yc = np.arange( 3333500,-3333500,-500)

    y_sub = xc[sub_idxs[2]:sub_idxs[3]]
    x_sub = yc[sub_idxs[0]:sub_idxs[1]]
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

    surf_interp = RGI(
        (y_sub, x_sub), surf_sub.T,
        bounds_error=False, fill_value=np.nan
    )

    bed_interp = RGI(
        (y_sub, x_sub), bed_sub.T,
        bounds_error=False, fill_value=np.nan
    )

    return {
        "surf_sub": surf_sub,
        "bed_sub": bed_sub,
        "X_sub": X_sub,
        "Y_sub": Y_sub,
        "surf_interp": surf_interp,
        "bed_interp": bed_interp
    }
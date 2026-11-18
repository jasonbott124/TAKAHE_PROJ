# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 20:03:10 2026

@author: Jason
"""

'''
LAYER ANALYSIS MASTER SCRIPT

Usage: should be used to run all layer analysis functions. 
Should NOT have big sections of noted-out code. S
hould have class inputs and those classes should be 
reused for everything (i.e. change the arguments in just one location)

Desired workflow is:
    1) Upload layers from various PSTs located within seperate folders
    2) Process layers into numpy arrays for plotting and examination
    3) Use pre-built functions to make calculations and spatial correlations (fold_maxima, shortening,
                                                                               --> frac_depth_plot, etc.)
    4) Plot Various PSTs together (max 3) on one large figure (i.e. X47a, Y31b, and NEGIS)
        a. Make sure the vertical and horizontal scales are SHARED between the plotted PSTs
        b. Make sure scale on other graphs are consistent too!
        c. age labels on layers would be useful, may be worth just plotting age-dated layers for now
    
    4.5) Plot the PST graphs seperately for use later, could be worth building into the functions or building here...either works. 
         Though, this script should be super simple plug-and-play, so let's try the former.
         
    Input: Folders
    
    Code: Specify Desired Analysis Zones (xlims) and what you 
    want to analyize (shortening, minima/maxima, vel/strain correlation, etc.)
    
    Output: Graphs and Dataframes Containing useful info

'''

# %% IMPORTS

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

import config  # central data/output paths (config.py)
import os
import sys
# sys.path.append(config.REPO_DIR)

import correct_layers as cl
from bm3_upload import load_bedmap3
from Mv2_uploader import Mv2_upload
import Velocity_Normal_Vectors as vnv
from layer_correlation_plot import layer_corr_plot_multiPST
from data_uploader_no_topo import *

# %% LOAD BEDMAP3

bm3_surf_path = config.BM3_SURF
bm3_bed_path = config.BM3_BED

bm3 = load_bedmap3(bm3_bed_path, bm3_surf_path, [6700,8400,3100,5200])

# %% LOAD Mv2 SURFACE VELOCITY

Mv2_path = config.MV2_VEL

Mv2_vel = Mv2_upload(Mv2_path, bm3)

# %% Upload and Correct Layers

# Ideally this should be the only thing to edit by the user in this script
PST_folder_list = { # MAX 3 (for now)
    "X47a": config.p(r"Margin Layers/X47a_Fixed_Maybe"),
    "Y31b": config.p(r"Margin Layers/Y31b Layers"),
    "Y17a": config.p(r"Margin Layers/Y17a Layers - NO BSB")
    }

# Produces a dataframe for layers in each PST
PST_layers = cl.produce_layer_df(PST_folder_list)

# Corrected Layers to their Absolute Elevation - calculation subject to change
PST_layers = cl.correct_layers_to_absolute_elevation(PST_layers, bm3["surf_interp"], bm3["bed_interp"])

FilesByPST = {}
for pst_name, layers in PST_layers.items():
    FilesByPST[pst_name] = cl.auto_pick_surface_bed_and_base(layers, base_choice="surface")
    print(FilesByPST[pst_name], "\n")
    
PST_layers = cl.layer_absdist_from_base(PST_layers,FilesByPST)

# %% USER DEFINED LIMITS

ArgumentsByPST = cl.pick_limits_by_pst(PST_layers,FilesByPST)

for pst, args in ArgumentsByPST.items():
    print(pst, args.limits_abs)

# %% ANALYSIS (PLOTS, GRAPHS, CALCS, ETC.)

Strains = vnv.Strain_Deliverer()

from types import SimpleNamespace

for pst in PST_layers.keys():
    
    abs_limits = ArgumentsByPST[pst].limits_abs
    
    ArgumentsByPST[pst] = SimpleNamespace(

        # ---- plotting ----
        depth="Elevation",              # or "Fractional"
        plot_style="line",
        point_size=10,
        vel_lims = (0, 120),
        strain_lims = (-0.1, 0.1),       
        melt_lims = (0, 30),
        layer_ylim = (-2000, 2000),

        # ---- velocity → melt ----
        low_thresh=10.0,
        upper_thresh=60.0, # Thresholds for sticky/sliding

        # ---- melt model ----
        melt_mode="deterministic",  # "statistical" or "deterministic"

        # ---- ice / basal params ----
        Bo=1.2e5,
        m=3.0,
        n=3.0, # Ice Flow Parameters that I want to keep constant

        # ---- MC placeholders (statistical only) ----
        N=100,
        Thickness=None,
        surf_vel=None,
        theta=None,
        limits_abs = abs_limits
        
    )

    
class Data:
    
    Vel = Mv2_vel
    Corrected_PST_list = PST_layers
    
class Interpolators:
    
    real_bed_interp = bm3["bed_interp"]
    real_surf_interp = bm3["surf_interp"]
    
# %% LAYER CORRELATION PLOT

layer_corr_plot_multiPST(FilesByPST, Interpolators, ArgumentsByPST, Data, Strains, 
                         global_vel_lims = (0, 120),
                         global_melt_lims = (0, 20),
                         global_strain_lims = (-0.08,0.08))

# %%

Data = data_uploader_no_topo(data_args = ["Accumulation"])

# %% VERTICAL VELOCITY PROFILE

import Vertical_Velocity_Profile as vp

PST_layers_new = vp.vertical_velocity_profile(PST_layers, 
                                              FilesByPST, Interpolators, 
                                              ArgumentsByPST, nz = 1000)






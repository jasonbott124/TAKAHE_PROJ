# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:20:28 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import numpy as np

def compare_strdip(A, B):
    """
    Compare corresponding rows of two 2D arrays A and B.
    Print rows (with index) that do not match exactly.
    """
    # Convert to numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Ensure same shape
    # if A.shape != B.shape:
    #     raise ValueError(f"Arrays must have the same shape, got {A.shape} and {B.shape}")
    
    repeat_or_not = np.zeros(A.shape[0], dtype=bool)
    
    # Loop through rows
    for i in range(A.shape[0]):
        
        if i == 0:
            print('i = 0')
        
        for j in range (B.shape[0]):
            
            if (A[i,0] == B[j,0]) and (A[i,1] == B[j,1]):
                
                repeat_or_not[i] = True
                
    idx = np.where(repeat_or_not == False)
    
    return idx, repeat_or_not
                
                

import pandas as pd

df1 = pd.read_excel(config.p(r"NE_Block_BreeCreek.xlsx"), header = None)
df2 = pd.read_excel(config.p(r"NE_Bree_Original.xlsx"), header = None)

d1 = np.array([df1[0],df1[1]]).T
d2 = np.array([df2[0],df2[1]]).T

idx, repeat = compare_strdip(d1,d2)

d1[idx]

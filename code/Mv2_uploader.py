# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:22:45 2026

@author: Jason
"""

def Mv2_upload(Mv2_path, bm3):
    
    import numpy as np
    import netCDF4 as nc
    from scipy.interpolate import RegularGridInterpolator as RGI

    Mv2 = nc.Dataset(Mv2_path)
    
    ylim = [np.min(bm3["X_sub"]), np.max(bm3["X_sub"])]  # X-coordinate limits
    xlim = [np.min(bm3["Y_sub"]), np.max(bm3["Y_sub"])]   # Y-coordinate limits
    
    x = Mv2['x'][:].data
    y = Mv2['y'][:].data
    
    xindex = np.where((x > xlim[0]) & (x < xlim[1]))
    yindex = np.where((y > ylim[0]) & (y < ylim[1]))
        
    x_filtered = x[xindex]
    y_filtered = y[yindex]
    
    Xc,Yc = np.meshgrid(x_filtered,y_filtered)

    vx_filtered = np.squeeze(Mv2['VX'][yindex][:,xindex].data)
    vy_filtered = np.squeeze(Mv2['VY'][yindex][:,xindex].data)
    errx_filtered = np.squeeze(Mv2['ERRX'][yindex][:,xindex].data)
    erry_filtered = np.squeeze(Mv2['ERRY'][yindex][:,xindex].data)
    stdx_filtered = np.squeeze(Mv2['STDX'][yindex][:,xindex].data)
    stdy_filtered = np.squeeze(Mv2['STDY'][yindex][:,xindex].data)
    
    erry_filtered[np.isnan(erry_filtered)] = 0 # Sets all NaN values to zero, we'll see if this fixes things
    errx_filtered[np.isnan(errx_filtered)] = 0
    
    fvx = RGI((x_filtered, y_filtered), vx_filtered.T, method='cubic', bounds_error=False)
    fvy = RGI((x_filtered, y_filtered), vy_filtered.T, method='cubic', bounds_error=False)
    
    ferrvx = RGI((x_filtered, y_filtered), errx_filtered.T, method='cubic', bounds_error=False)
    ferrvy = RGI((x_filtered, y_filtered), erry_filtered.T, method='cubic', bounds_error=False)
    
    fstdvx = RGI((x_filtered, y_filtered), stdx_filtered.T, method='cubic', bounds_error=False)
    fstdvy = RGI((x_filtered, y_filtered), stdy_filtered.T, method='cubic', bounds_error=False)
    
    v_sub = np.sqrt(vx_filtered**2 + vy_filtered**2)
    
    return {
        "x_filtered": x_filtered,
        "y_filtered": y_filtered,
        "Xc": Xc,
        "Yc": Yc,
        "vx": vx_filtered,
        "vy": vy_filtered,
        "errx": errx_filtered,
        "erry": erry_filtered,
        "surface_speed": v_sub,
        "fvx": fvx,
        "fvy": fvy,
    }
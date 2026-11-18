# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 14:21:00 2025

@author: Jason
"""

import config  # central data/output paths (config.py)
import os
import numpy as np
import pandas as pd
import xarray as xr

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
from Flowband_Functions import new_tif_to_array

def build_xy_from_extent_and_shape(extent_x, extent_y, shape):
    """
    Build coordinate arrays from map extent and resolution.
    """
    
    nx, ny = shape
    
    x = np.linspace(extent_x[0], extent_x[1], nx)
    y = np.linspace(extent_y[0], extent_y[1], ny)
    X, Y = np.meshgrid(x, y)
    
    return x, y, X, Y

def attach_coordinates(obj, x, y, X=None, Y=None):
    """
    Attach coordinate arrays to a GeoDataObject in-place.
    """
    obj.x_array = x
    obj.y_array = y
    obj.X = X
    obj.Y = Y

# ============================================================
# Data object container
# ============================================================

class GeoDataObject:
    
    def __init__(self, file_name, file_path, file_type,
                 array, x_array=None, y_array=None):

        self.file_name = file_name
        self.file_path = file_path
        self.file_type = file_type
        self.array = array
        self.x_array = x_array
        self.y_array = y_array

    def __repr__(self):
        shape = None if self.array is None else self.array.shape
        return f"<GeoDataObject {self.file_name} | {self.file_type} | {shape}>"


# ============================================================
# Loader function
# ============================================================

def load_non_velocity_non_topo_data(data_args=None):
    """
    Load selected datasets into a DataFrame of GeoDataObjects.

    Parameters
    ----------
    data_args : list[str] or None
        Dataset categories to load (e.g. ["Accumulation", "Geothermal"]).
        If None, all categories are loaded.

    Returns
    -------
    Data : pandas.DataFrame
        Columns are dataset categories.
        Rows are file names.
        Values are GeoDataObject instances.
    """

    base_dir = config.p(r"TAKAHE_data")

    # Normalize args
    if data_args is not None:
        if isinstance(data_args, str):
            data_args = [data_args]
        data_args = [arg.lower() for arg in data_args]

    exclude_keywords = [
        "vx", "vy", "vel", "velocity", "speed",
        "surf", "surface", "bed", "bedrock", "topo"
    ]

    # Object-holding DataFrame
    Data = pd.DataFrame(dtype=object)

    # --------------------------------------------------------
    # Pre-scan for progress reporting
    # --------------------------------------------------------
    selected_files = []

    for root, _, files in os.walk(base_dir):
        root_lower = root.lower()

        if data_args is not None:
            if not any(arg in root_lower for arg in data_args):
                continue

        for file in files:
            file_lower = file.lower()
            if any(k in file_lower for k in exclude_keywords):
                continue
            selected_files.append(os.path.join(root, file))

    n_selected = len(selected_files)

    print(f"\n📂 Files selected for parsing: {n_selected}\n")

    parsed_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # Main load loop
    # --------------------------------------------------------
    for root, _, files in os.walk(base_dir):

        root_lower = root.lower()

        if data_args is not None:
            if not any(arg in root_lower for arg in data_args):
                continue

        category = os.path.basename(root)

        if category not in Data.columns:
            Data[category] = pd.Series(dtype=object)

        for file in files:
            file_lower = file.lower()

            if any(k in file_lower for k in exclude_keywords):
                continue

            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                # ------------------ GeoTIFF ------------------
                if ext == ".tif":
                    array = new_tif_to_array(file_path)

                    obj = GeoDataObject(
                        file_name=os.path.splitext(file)[0],
                        file_path=file_path,
                        file_type="tif",
                        array=array,
                        # x_array=x,
                        # y_array=y
                    )
                    
                    print("\nWARNING: tif file does not contain coordinate information and must be georeferenced post upload!")

                    Data.at[obj.file_name, category] = obj

                # ------------------ NetCDF -------------------
                elif ext == ".nc":
                    ds = xr.open_dataset(file_path)

                    for var_name in ds.data_vars:
                        da = ds[var_name]

                        obj = GeoDataObject(
                            file_name=var_name,
                            file_path=file_path,
                            file_type="nc",
                            array=da.values,
                            x_array=da.coords.get("x", None),
                            y_array=da.coords.get("y", None)
                        )

                        Data.at[var_name, category] = obj

                    ds.close()

                else:
                    continue

                parsed_count += 1
                print(f"✅ [{parsed_count}/{n_selected}] Loaded: {file}")

            except Exception as e:
                failed_count += 1
                print(f"❌ Failed: {file_path}")
                print(f"   ↳ {e}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\nLoad summary")
    print("----------------")
    print(f"Successfully loaded: {parsed_count}")
    print(f"Failed:              {failed_count}")
    print(f"Total attempted:     {n_selected}")

    return Data


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    Data = load_non_velocity_non_topo_data(
        data_args=["Accumulation", "Geothermal"]
    )


def upload_measures_sv(Path, ylim, xlim):
    
    from scipy.interpolate import RegularGridInterpolator as RGI
    import numpy as np
    import netCDF4 as nc
    import numpy as np
    
    Mv2 = nc.Dataset(Path)

    # ylim = [np.min(bm3.X_sub), np.max(bm3.X_sub)]  # X-coordinate limits
    # xlim = [np.min(bm3.Y_sub), np.max(bm3.Y_sub)]   # Y-coordinate limits

    class Vel:
        
        raw_x = Mv2['x'][:].data
        raw_y = Mv2['y'][:].data
        
        xindex = np.where((raw_x > xlim[0]) & (raw_x < xlim[1]))
        yindex = np.where((raw_y > ylim[0]) & (raw_y < ylim[1]))
        
        x_filtered = raw_x[xindex]
        y_filtered = raw_y[yindex]
        
        Xc,Yc = np.meshgrid(x_filtered,y_filtered)

        vx_filtered = np.squeeze(Mv2['VX'][yindex][:,xindex].data)
        vy_filtered = np.squeeze(Mv2['VY'][yindex][:,xindex].data)
        errx_filtered = np.squeeze(Mv2['ERRX'][yindex][:,xindex].data)
        erry_filtered = np.squeeze(Mv2['ERRY'][yindex][:,xindex].data)
        stdx_filtered = np.squeeze(Mv2['STDX'][yindex][:,xindex].data)
        stdy_filtered = np.squeeze(Mv2['STDY'][yindex][:,xindex].data)
    
        erry_filtered[np.isnan(erry_filtered)] = 0 # Sets all NaN values to zero, we'll see if this fixes things
        errx_filtered[np.isnan(errx_filtered)] = 0
    
        fvx = RGI((x_filtered, y_filtered), vx_filtered.T, method='linear', bounds_error=False)
        fvy = RGI((x_filtered, y_filtered), vy_filtered.T, method='linear', bounds_error=False)
    
        ferrvx = RGI((x_filtered, y_filtered), errx_filtered.T, method='cubic', bounds_error=False)
        ferrvy = RGI((x_filtered, y_filtered), erry_filtered.T, method='cubic', bounds_error=False)
    
        fstdvx = RGI((x_filtered, y_filtered), stdx_filtered.T, method='cubic', bounds_error=False)
        fstdvy = RGI((x_filtered, y_filtered), stdy_filtered.T, method='cubic', bounds_error=False)
    
        v_sub = np.sqrt(vx_filtered**2 + vy_filtered**2)
        
        # --- Flow magnitude and unit vectors ---
        speed = np.sqrt(vx_filtered**2 + vy_filtered**2)
        eps = 1e-12
        mask = speed < eps

        ux = np.full_like(vx_filtered, np.nan, dtype=float)
        uy = np.full_like(vy_filtered, np.nan, dtype=float)
        ux[~mask] = vx_filtered[~mask] / speed[~mask]
        uy[~mask] = vy_filtered[~mask] / speed[~mask]

        # Optional perpendicular (normal) vectors, rotated +90°
        nx = -uy
        ny = ux
        
        dy = 450
        dx = 450
        
        normal_angles = np.arctan(ny/nx)
        dudx = np.gradient(vx_filtered, dy, dx)[0]
        dvdy = np.gradient(vy_filtered, dy, dx)[1]
        dvdx = np.gradient(vy_filtered, dy, dx)[0]
        dudy = np.gradient(vx_filtered, dy, dx)[1]
        dev_strain = 0.5*(dvdx + dudy)
        
        a = normal_angles
        lon_strain = dudx*(np.cos(a)**2) + 2*dev_strain*np.cos(a)*np.sin(a) + dvdy*(np.sin(a)**2)
        lat_strain = dudx*(np.sin(a)**2) - 2*dev_strain*np.cos(a)*np.sin(a) + dvdy*(np.cos(a)**2)
        shear_strain = (dvdy - dudx)*np.cos(a)*np.sin(a) + dev_strain*(np.cos(a)**2 - np.sin(a)**2)
        eff_strain = 0.5*np.sqrt(dudx**2 + 2*dev_strain**2 + dvdy**2)
        
    return Vel
    

def upload_bedmap3(bm3_bed_path, bm3_surf_path, xidx, yidx):
    
    import sys
    # (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
    import Flowband_Functions as ff
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI
    
    class bm3:

        bed = ff.new_tif_to_array(bm3_bed_path)
        surf = ff.new_tif_to_array(bm3_surf_path)
        thickness = surf - bed
        
        # xidx = [7000,8500]
        # yidx = [3000,4500]

        bed_sub = bed[xidx[0]:xidx[1],yidx[0]:yidx[1]].astype(float)
        surf_sub = surf[xidx[0]:xidx[1],yidx[0]:yidx[1]].astype(float)
        surf_sub[surf_sub < -5000] = np.nan

        thk_sub = surf_sub - bed_sub

        xc = np.arange(-3333500,3333500,500)
        yc = np.arange(3333500,-3333500,-500)

        [Xc, Yc] = np.meshgrid(xc,yc)
        
        y_sub = xc[yidx[0]:yidx[1]]
        x_sub = yc[xidx[0]:xidx[1]]
        
        X_sub, Y_sub = np.meshgrid(x_sub,y_sub)
        
        surf_interp = RGI(
        (x_sub, y_sub),          # Axis order is (Y, X)
        surf_sub.T,              # Must match shape (len(Y), len(X))
        method='linear',         # or 'cubic'
        bounds_error=False,
        fill_value=np.nan
        )
        
        bed_interp = RGI(
        (x_sub, y_sub),          # Axis order is (Y, X)
        bed_sub.T,               # Must match shape (len(Y), len(X))
        method='linear',         # or 'cubic'
        bounds_error=False,
        fill_value=np.nan
        )

        thk_interp = RGI(
                (x_sub, y_sub),          # Axis order is (Y, X)
                thk_sub.T,               # Must match shape (len(Y), len(X))
                method='linear',         # or 'cubic'
                bounds_error=False,
                fill_value=np.nan
                )
        
    return bm3
"""
load_TAKAHE_datasets.py

Loads non-velocity, non-topography datasets from the TAKAHE data directory
into a pandas DataFrame of objects grouped by dataset category.
"""

import config  # central data/output paths (config.py)
import os
import numpy as np
import pandas as pd
import xarray as xr

import sys
# (hard-coded sys.path entry removed: Flowband_Functions now lives in this repo)
from Flowband_Functions import new_tif_to_array


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

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 16:11:22 2025

@author: Jason
"""

# Change projection

import config  # central data/output paths (config.py)
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ----------------------------
# User settings
# ----------------------------

input_file = config.p(r"rasters_COP30/output_hh.tif")          # your .hh raster
output_file = config.p(r"kilimanjaro_utm.tif")   # output GeoTIFF in meters

source_crs = "EPSG:4326"         # current CRS (lat/lon, degrees)

# Choose the target CRS depending on your location:
# Kilimanjaro → EPSG:32737  (UTM 37S, meters)
# Mauna Loa   → EPSG:32605  (UTM 5N, meters)
target_crs = "EPSG:32737"

# Optional: choose a specific output resolution in meters (None = auto)
target_resolution = None
# Example: target_resolution = (30, 30)  # 30 m pixels

input_csv = config.p(r"ASE_tpros/ASE5_rad_data_foc.csv")
output_csvr = config.p(r"ASE_tpros/ASE5_rad_data_foc_EPS1301.csv")


# ----------------------------
# Reprojection
# ----------------------------

with rasterio.open(input_file) as src:
    # Compute transform & shape in the target CRS
    transform, width, height = calculate_default_transform(
        src.crs or source_crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds,
        resolution=target_resolution
    )

    # Prepare metadata for the output file
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "driver": "GTiff"  # Make sure output is a TIFF
    })

    # Create the output raster and reproject pixel values
    with rasterio.open(output_file, "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=source_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear   # Recommended for DEM
        )

print(f"Reprojection complete! Saved to: {output_file}")

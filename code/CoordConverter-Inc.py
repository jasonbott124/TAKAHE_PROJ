# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:23:29 2026

@author: Jason
"""

# PS90 to PS71 converter

def PS90toPS71(input_x, input_y):

    import pandas as pd
    from pyproj import Transformer
    
    # 1. Choose your polar projection (e.g., EPSG:3413 for North Pole)
    # Use "always_xy=True" to consistently accept (longitude, latitude) input order
    transformer = Transformer.from_crs("epsg:3031", "epsg:4326", always_xy=True)
    
    # 2. Convert a geographic point (Lon, Lat) to Polar Stereographic coordinates (X, Y in meters)
    x, y = input_x, input_y  # Directly at the North Pole
    lon, lat = transformer.transform(x, y)
    
    print(f"Projected X: {x}, Y: {y}")
    
    # 3. Perform the inverse transformation (X, Y back to Lon, Lat)
    inverse_transformer = Transformer.from_crs("epsg:3413", "epsg:4326", always_xy=True)
    lon_out, lat_out = inverse_transformer.transform(x, y)
    
    print(f"Geographic Lon: {lon_out}, Lat: {lat_out}")
    
    # %%
    
    # 1. Choose your PS71 destination (e.g., EPSG:3995 for Arctic PS71)
    # EPSG:3995 sets lat_ts=71 (latitude of standard parallel) and lon_0=0
    destination_crs = "epsg:3031" 
    
    # 2. Set up the transformer from WGS84 (lat/lon) to PS71
    transformer = Transformer.from_crs("epsg:4326", destination_crs, always_xy=True)
    
    # 3. Transform a point (Lon, Lat) 
    # Example: Longitude 15.0, Latitude 78.0 (Svalbard region)
    # lon_out, lat_out = 15.0, 78.0  
    misc_pts_X, misc_pts_Y = transformer.transform(lon_out, -lat_out)
    
    # print(f"PS71 Projected X: {misc_pts_X:.2f} meters, Y: {misc_pts_Y:.2f} meters")

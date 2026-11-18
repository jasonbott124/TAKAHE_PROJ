# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:29:58 2024

@author: Jason
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import cmocean
import PIL
from scipy.interpolate import RegularGridInterpolator as RGI

# Functions for Ice-Flowband Tracing Code

def plot_Surface_3D(surf,X,Y,x,y,DataType,cmapv,exag=0.01,contours=True,plot2D=True):
    
    # from plotly.offline import init_notebook_mode
    # init_notebook_mode()
    
    import plotly.io as pio
    pio.renderers.default = 'browser'
    
    # Convert cmocean colormap to Plotly format
    def cmocean_to_plotly(cmap, num_colors=255):
        return [[i/(num_colors-1), f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'] 
                for i, (r, g, b, _) in enumerate(cmap(np.linspace(0, 1, num_colors)))]
    
    if DataType == 'Bed':
        # Convert cmocean 'thermal_r' colormap to Plotly format
        cmap = cmocean_to_plotly(cmocean.cm.topo)
        title = 'Bed Elevation (m)'
        # cmin = np.min(surf)
        # cmax = -cmin
        max_abs = max(abs(np.min(surf)), abs(np.max(surf)))
        cmin, cmax = -max_abs, max_abs 
        
    elif DataType == 'Surface':
        # Convert cmocean 'thermal_r' colormap to Plotly format
        cmap = 'turbo'
        title = 'Surface Elevation (m)'
        # cmin = np.min(surf)
        # cmax = np.max(surf)
        cmin = 0
        cmax = 3000
        
    elif DataType == 'Velocity':
        cmap = 'turbo'
        title = 'Surface Velocity (m$a^-1$)'
        cmin = np.min(cmapv)
        cmax = np.max(cmapv)
        
    else:
        return 'Error: No DataType Selected, please input "Bed" or "Surface" as DataType variable'

    fig = go.Figure(data=[
        go.Surface(
            z=surf,
            x=X,
            y=Y,
            surfacecolor=cmapv,   # <--- color determined by cmapv
            colorscale=cmap,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title='Color Mapping')
        )
    ])

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        scene = dict(
            xaxis_title='Eastings [m]',
            yaxis_title='Northings [m]',
            zaxis_title=title,
            aspectratio=dict(x=1.3, y=1.0, z=exag)
        ),
        autosize=True,
           width=1920,   # or bigger depending on your screen resolution
           height=1080,
        margin=dict(l=45, r=50, b=65, t=100)
    )

    # Add color bar
    fig.update_layout(coloraxis_colorbar=dict(
        title=title,
        tickvals=[3],
        ticks="outside"
    ))

    if contours == True:
        fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                      highlightcolor="black", project_z=True))

    # Show the plot
    # fig.show()

    return fig
    
    if plot2D:
        plt.figure(figsize=(20, 8))
        plt.imshow(cmapv, cmap=cmap, vmin=cmin, vmax=cmax,
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label='Color Mapping Value')
        plt.title('2D Map (color from cmapv)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
        
# Convert cmocean colormap to Plotly format
def cmocean_to_plotly(cmap, num_colors=255):
    return [[i/(num_colors-1), f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'] 
        for i, (r, g, b, _) in enumerate(cmap(np.linspace(0, 1, num_colors)))]
        
        
def plot_Velocity(surf,velocity,X,Y,v_exag,cmocean_map,plot2D=True,contours=True):
    
    import plotly.io as pio
    pio.renderers.default = 'browser'
    
    # from plotly.offline import init_notebook_mode
    # init_notebook_mode()

    # Create an empty figure
    fig = go.Figure()

    # Convert cmocean 'thermal_r' colormap to Plotly format
    if not isinstance(cmocean_map, str):
        thermal_r_colorscale = cmocean_to_plotly(cmocean_map)
    else:
        thermal_r_colorscale = cmocean_map

    # Add the velocity magnitude skin using its own grid (without interpolation)
    fig.add_trace(go.Surface(
        z=surf,  # Use the same 'z' values from the sub surface for alignment
        x=X,  # X coordinates for velocity data (original velocity grid)
        y=Y,  # Y coordinates for velocity data (original velocity grid)
        surfacecolor=velocity,  # Use the original velocity magnitude for coloring
        colorscale= thermal_r_colorscale,  # Colorscale for the velocity data
        opacity=1.0,  # Make the velocity skin slightly transparent to overlay on the surface
        colorbar=dict(title='Velocity Magnitude (m/year)'),
        cmin=0,  # Minimum value for color scaling
        cmax=1100# Show the color bar for velocity magnitude
    ))

    fig.update_layout(
        title='Interpolated 3D Surface with Velocity Magnitude',  # Final title
        scene=dict(
            xaxis_title='X Coordinate (meters)',  # Consistent axis labels
            yaxis_title='Y Coordinate (meters)',
            zaxis_title='Elevation/Data Values',
            aspectratio=dict(x=1.5, y=1.0, z=v_exag),  # Adjust z ratio for vertical exaggeration
            zaxis=dict(title='Z Axis'),
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
        ),
        width=1920,   # or bigger depending on your screen resolution
        height=1080,
        # margin=dict(l=45, r=50, b=65, t=100),  # Final margins
        autosize=True,  # Make the figure auto-resize
        # scene_camera=dict(eye=dict(x=1.75, y=1.75, z=0.75)),  # Adjust camera perspective
        coloraxis_colorbar=dict(
            title="Surface Velocity [m/yr]",  # Color bar title
            tickvals=[4],
            ticks="outside"
        )
    )

    if contours == True:

        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="white", project_z=True))

    if plot2D == True:
        # Visualize the data using Matplotlib
        plt.figure(figsize=(20, 8))
        plt.imshow(velocity, cmap='viridis',vmin=np.min(velocity),vmax=np.max(velocity))
        plt.colorbar(label='Surface Velocity (m/yr)')
        plt.title('Mv2 Surface Velocity')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
    # Show the plot
    fig.show()
    
    return fig


def binary_reader(hdr_file,flt_file,plot=True):
    
    # Read the .hdr file
    def read_hdr(hdr_file):
        hdr_data = {}
        with open(hdr_file, 'r') as file:
            for line in file:
                key, value = line.split()
                hdr_data[key.lower()] = value
        return hdr_data

    # Read the .hdr file to extract metadata
    hdr_data = read_hdr(hdr_file)

    # Extract necessary metadata (ncols, nrows, byteorder)
    ncols = int(hdr_data['ncols'])
    nrows = int(hdr_data['nrows'])
    byteorder = hdr_data['byteorder'].lower()  # LSBFIRST means little-endian
    nodata_value = float(hdr_data['nodata_value'])

    # Set data type to float32 since the .flt file stores floating-point data
    dtype = np.float32

    # Read the .flt file using NumPy with correct byte order
    if byteorder == 'lsbfirst':
        dtype = np.dtype('<f4')  # Little-endian 32-bit float
    else:
        dtype = np.dtype('>f4')  # Big-endian 32-bit float

    # Read the binary .flt file
    raster_data = np.fromfile(flt_file, dtype=dtype).reshape((nrows, ncols))

    # Replace NODATA values with NaN for easier processing/plotting
    raster_data[raster_data == nodata_value] = np.nan
    
    xllcorner = float(hdr_data['xllcorner'])
    yllcorner = float(hdr_data['yllcorner'])
    cellsize = float(hdr_data['cellsize'])

    x = np.linspace(xllcorner, xllcorner + (ncols - 1) * cellsize, ncols)
    y = np.linspace(yllcorner, yllcorner + (nrows - 1) * cellsize, nrows)

    if(plot==True):
        # Visualize the data using Matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(raster_data, cmap='ocean' 
                   ,extent=[
            float(hdr_data['xllcorner']),
            float(hdr_data['xllcorner']) + ncols * float(hdr_data['cellsize']),
            float(hdr_data['yllcorner']),
            float(hdr_data['yllcorner']) + nrows * float(hdr_data['cellsize']),
        ]
            )
        plt.colorbar(label='Surface Elevation (m asl)')
        plt.title('Bedmap v2 {}')
        plt.xlabel('Eastings (m)')
        plt.ylabel('Northings (m)')
        plt.grid(True)
        plt.show()
    
    return raster_data, hdr_data, ncols, nrows, x, y

def get_gradient(ix, iy, thk_interpolator, scf = 10):
    # Compute central difference for gradient with scaled step size
    h = thk_interpolator((ix, iy))  # Ice thickness, needs to be created
    dz_dx = (get_z_value(ix + h * scf, iy) - get_z_value(ix - h * scf, iy)) / (2 * h * scf)
    dz_dy = (get_z_value(ix, iy + h * scf) - get_z_value(ix, iy - h * scf)) / (2 * h * scf)
    return np.sqrt(dz_dx**2 + dz_dy**2)

def get_z_value(ix, iy, z_interpolator):
    # Find the closest indices for the x and y coordinates
    return z_interpolator((ix, iy))

        
def flowline_tracer(starting_x, starting_y, fvx, fvy, ferrvx, ferrvy, nsteps, step, fig, realizations, Show_P_Bar=True):
    
    from tqdm import tqdm  # This function will display a progress bar
    from plotly.figure_factory import create_colorscale as cmocean_to_plotly
    import cmocean
    import random
    
    starting_points = [starting_x, starting_y]
    
    # xlist_final = []
    # ylist_final = []
    # zlist_final = []
    
    i = 0
    while i < len(realizations):
        xlist = []
        ylist = []
        zlist = []
        ss = []
        ix, iy = starting_points[0][i], starting_points[1][i]
        iz = get_z_value(ix, iy)
        
        loop = tqdm(range(nsteps)) if Show_P_Bar else range(nsteps)
        for aa in loop:
            xlist.append(ix)
            ylist.append(iy)
            zlist.append(iz)
            ss.append(get_gradient(ix, iy))
            ivx = fvx((ix, iy)) + random.uniform(-ferrvy((ix,iy)),ferrvy((ix,iy)))
            ivy = fvy((ix, iy)) + random.uniform(-ferrvy((ix,iy)),ferrvy((ix,iy)))
            vt = (ivx**2 + ivy**2)**0.5
            ystep = step * ivy / vt
            xstep = step * ivx / vt
            ix -= xstep
            iy -= ystep
            iz = get_z_value(ix, iy)
            
            if aa == nsteps - 1:
                # Convert cmocean colormap to a Plotly-compatible colorscale
                colormap = cmocean_to_plotly(cmocean.cm.haline)
                
                # Add the flowline as a 3D scatter trace with color based on ss
                fig.add_trace(go.Scatter3d(
                    x=xlist,  # Flowline X coordinates
                    y=ylist,  # Flowline Y coordinates
                    z=zlist,  # Flowline Z coordinates (e.g., elevation)
                    mode='lines',
                    line=dict(
                        width=4,
                        color=ss,  # Use the actual ss values for coloring
                        colorscale=colormap,  # Choose a colorscale
                        cmin=0.0,  # Set the minimum of ss for the color range
                        cmax=0.2   # Set the maximum of ss for the color range
                    ),
                    name=f'Flowline {i+1}'
                ))
                
                print(f'Flowline {i + 1} complete')
                i += 1

    print("All flowlines completed")

    
    
    # Adjust layout for better visualization
    fig.update_layout(
        title='3D Surface with Flowlines and Velocity Magnitude',
        scene=dict(
            xaxis_title='X Coordinate (meters)',
            yaxis_title='Y Coordinate (meters)',
            zaxis_title='Elevation/Data Values',
        ),
        width=1400,
        height=1000,
        scene_camera=dict(
            eye=dict(x=1.75, y=1.75, z=0.75)  # Optionally adjust the camera perspective
        )
    )
    
    # Show the plot
    fig.show()
    
def tif_to_array(image_path,x=[],y=[],set_extents=False):
    
    from PIL import Image
    import numpy as np

    # Load the TIFF image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    image_array = np.nan_to_num(image_array, nan=0)
    
    if set_extents == True:
    
        extent_x = np.linspace(np.min(x),np.max(x),len(image_array[:,0]))
        extent_y = np.linspace(np.min(y),np.max(y),len(image_array[0,:]))
        
        print(np.shape(image_array))
        print(np.shape(x),np.shape(y))
    
        X,Y = np.meshgrid(extent_x,extent_y)
        
        return image_array, X, Y, extent_x, extent_y

    else:
        return image_array
    


def universal_file_opener(file_path,file_type,hdr='none',flt='none'):
    
    if(file_type=='grd'):
    
        import xarray as xr
        
        # Load the .grd file
        grd_data = xr.open_dataset(file_path)
        
        # Access the variable
        print(grd_data)  # This will show the structure of the dataset
        BM_data = grd_data['z']  # Replace 'variable_name' with the actual name of the data variable
        x_data = grd_data['x']
        y_data = grd_data['y']
        
        # Convert to numpy array if needed
        gdata = BM_data.values
        x = x_data.values
        y = y_data.values
        
        return gdata, x, y
    
    elif(file_type=='bin'):
        raster_data, hdr_data, ncols, nrows, x, y = binary_reader(hdr,flt)
        
    elif(file_type=='nc'):
        import netCDF4 as nc
        dataset = nc.Dataset(file_path,'r')
        # Get the variable names
        variable_names = dataset.variables.keys()

        # Print the variable names
        for var_name in variable_names:
            print(var_name)
        
        return dataset
    
    elif(file_type=='tif'):
        
        image_array = tif_to_array(file_path)
        return image_array
    
    
def new_tif_to_array(image_path,x=[],y=[],set_extents=False):
    
    from PIL import Image
    import numpy as np

    # Load the TIFF image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    image_array = np.nan_to_num(image_array, nan=0)
    
    if set_extents == True:
    
        extent_x = np.linspace(np.min(x),np.max(x),len(image_array[:,0]))
        extent_y = np.linspace(np.min(y),np.max(y),len(image_array[0,:]))
        
        print(np.shape(image_array))
        print(np.shape(x),np.shape(y))
    
        X,Y = np.meshgrid(extent_x,extent_y)
        
        return image_array, X, Y, extent_x, extent_y

    else:
        return image_array
    
    
def build_interpolator(X,Y,Data):
    
    interpolator = RGI((X[0, :], Y[:, 0]), Data.T, method='linear', bounds_error=False, fill_value=None)
    
    return interpolator


def MuldoonLeuro_csv_reader(folder):
    
    import os
    import pandas as pd

    # --- STORAGE ---
    all_pst_layers = {}  # dictionary of arrays keyed by filename
    file_name_list = []
    
    for file in os.listdir(folder):
        
        if not file.endswith(".csv"):
            continue
    
        filepath = os.path.join(folder, file)
    
        # Read horizon file
        df = pd.read_csv(filepath)
    
        # Convert lat/lon -> projected coords
        east, north = df['x'], df['y']
    
        # Find Layer ELevation and Store EPS 1301 Locations
        x_array = np.array(east)
        y_array = np.array(north)
        layer_depth = df["IRHdepth_m"].values
        surface_elev = df['thk_m'] + df['bedelv_mABSL']
        layer_elev = surface_elev - layer_depth
    
        # Compute cumulative distance along track
        dx = np.diff(x_array)
        dy = np.diff(y_array)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        x_position = np.concatenate([[0], np.cumsum(segment_lengths)])
    
        all_pst_layers[file] = {
            "x_array": x_array,
            "y_array": y_array,
            "z_array": layer_elev,
            "x_position": x_position,
        }
        
        file_name_list.append(file)
        
    return all_pst_layers, file_name_list
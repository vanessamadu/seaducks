# seaducks/data_processing/derived_quantities.py
from seaducks.utils import diff1d,herald
import xarray as xr
import os
import glob
from datetime import datetime
import numpy as np

# ----------------- SST gradient ------------------ #
def sst_gradient_pointwise(sst_array: xr.DataArray, coord: tuple, time_val: float) -> tuple:
    
    # metadata
    h = 0.05                       # degrees
    earth_radius = 6371
    h = np.deg2rad(h)*earth_radius # convert to metres

    lat_val,lon_val = coord
    # find sst values near coord
    lat_neighbours = [lat_val+ii*h for ii in np.arange(-2,3,1)]
    lon_neighbours = [lon_val+jj*h for jj in np.arange(-2,3,1)]

    sst_x_neighbours = sst_array.sel(latitude = lat_neighbours, time=time_val, longitude = lon_val, method='nearest').values
    sst_y_neighbours = sst_array.sel(latitude=lat_val, time=time_val,longitude = lon_neighbours, method = 'nearest').values

    return (diff1d(sst_x_neighbours,h)[2],diff1d(sst_y_neighbours,h)[2]) # return centre values



def sst_gradient(SST_array: xr.DataArray) -> tuple:
    
    # metadata
    h = 0.05                       # degrees
    earth_radius = 6371
    h = np.deg2rad(h)*earth_radius # convert to metres

    lat = SST_array['latitude']
    lon = SST_array['longitude']
    t_array = SST_array['time']

    # initialisation
    SST_grad_array_x = xr.zeros_like(SST_array)
    SST_grad_array_y = xr.zeros_like(SST_array)

    for tt, t_val in enumerate(t_array):
        for ii, lat_val in enumerate(lat):
            sst_at_lat = SST_array.sel(latitude=lat_val, time=t_val)
            SST_grad_array_x.loc[dict(time=t_val, latitude=lat_val)] = diff1d(sst_at_lat.values, h)
            # print and log progress
            if ii % 50 == 0:
                herald(f"Latitude progress: {ii * 100 / len(lat):.2f}% complete")

        for jj, lon_val in enumerate(lon):
            sst_at_lon = SST_array.sel(longitude=lon_val, time=t_val)
            SST_grad_array_y.loc[dict(time=t_val, longitude=lon_val)] = diff1d(sst_at_lon.values, h)
            # print and log progress
            if jj % 50 == 0:
                print(f"Longitude progress: {jj * 100 / len(lon):.2f}% complete")

        herald(f"Time progress: {tt * 100 / len(t_array):.2f}% complete")
    return SST_grad_array_x,SST_grad_array_y

def sst_gradient_to_da(input_directory,file_pattern,output_directory,output_filename):

    sst_files = glob.glob(os.path.join(input_directory, file_pattern))

    all_gradients_x = []
    all_gradients_y = []

    for sst_file in sst_files:
        herald(f"Processing {sst_file}")
        sst = xr.open_dataset(sst_file)['analysed_sst']
        
        sst_gradient_x, sst_gradient_y = sst_gradient(sst)

        all_gradients_x.append(sst_gradient_x)
        all_gradients_y.append(sst_gradient_y)

    # Concatenate all gradients along the time dimension
    combined_gradients_x = xr.concat(all_gradients_x, dim='time')
    combined_gradients_y = xr.concat(all_gradients_y, dim='time')

    # Create data arrays
    combined_gradients_x.name = 'sst_gradient_x'
    combined_gradients_y.name = 'sst_gradient_y'
    # create dataset
    sst_gradient_ds = xr.Dataset(
        {
            'sst_gradient_x':combined_gradients_x,
            'sst_gradient_y':combined_gradients_y
        }
    )
    herald("dataset created successfully")
    # set attributes for the gradient variables
    sst_gradient_ds['sst_gradient_x'].attrs = {
        'units': 'K/km',
        'long_name': 'SST Gradient in X direction'
    }
    sst_gradient_ds['sst_gradient_y'].attrs = {
        'units': 'K/km',
        'long_name': 'SST Gradient in Y direction'
    }

    # Set global attributes
    sst_gradient_ds.attrs = {
        'title': 'Sea Surface Temperature Gradients',
        'source': 'Derived from SST data',
        'history': 'Created on ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'references': '[ADD REFERENCES]'
    }
    
    herald("new attributes assigned successfully")
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Write to NetCDF file
    file_path = os.path.join(output_directory, output_filename)
    sst_gradient_ds.to_netcdf(file_path)

    herald(f"data successfully written to {file_path}")

# ---------------- interpolation ------------------- #



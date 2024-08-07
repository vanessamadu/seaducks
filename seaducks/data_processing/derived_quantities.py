# seaducks/data_processing/derived_quantities.py
from seaducks.utils import diff1d,herald
import xarray as xr
import os
from datetime import datetime

# ----------------- SST gradient ------------------ #

def sst_gradient(SST_array: xr.DataArray) -> tuple:
    
    # metadata
    h = 0.05
    lat = SST_array['latitude']
    lon = SST_array['longitude']
    t_array = SST_array['time']

    # initialisation
    SST_grad_array_x = xr.zeros_like(SST_array)
    SST_grad_array_y = xr.zeros_like(SST_array)

    for tt in t_array:
        for ii in lat:
            sst_at_lat = SST_array.sel(latitude = ii,time=tt)
            SST_grad_array_x.loc[dict(time=tt, latitude=ii)] = diff1d(sst_at_lat.values, h)
            print(f"lat:{ii.values} complete")
        
        for jj in lon:
            sst_at_lon = SST_array.sel(longitude = jj,time=tt)
            SST_grad_array_y[tt,:,jj] = diff1d(sst_at_lon,h)
            print(f"lon:{lon[jj]} complete")
    return SST_grad_array_x,SST_grad_array_y

def sst_gradient_to_da(sst,output_directory,output_filename):

    sst_gradient_x, sst_gradient_y = sst_gradient(sst)

    # create data arrays
    sst_gradient_x.name='sst_gradient_x'
    sst_gradient_y.name='sst_gradient_y'

    # create dataset
    sst_gradient_ds = xr.Dataset(
        {
            'sst_gradient_x':sst_gradient_x,
            'sst_gradient_y':sst_gradient_y
        }
    )

    # set attributes for the gradient variables
    sst_gradient_ds['sst_gradient_x'].attrs = {
        'units': 'K/m',
        'long_name': 'SST Gradient in X direction'
    }
    sst_gradient_ds['sst_gradient_y'].attrs = {
        'units': 'K/m',
        'long_name': 'SST Gradient in Y direction'
    }

    # Set global attributes
    sst_gradient_ds.attrs = {
        'title': 'Sea Surface Temperature Gradients',
        'source': 'Derived from SST data',
        'history': 'Created on ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'references': '[ADD REFERENCES]'
    }
    

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Write to NetCDF file
    file_path = os.path.join(output_directory, output_filename)
    sst_gradient_ds.to_netcdf(file_path)

    herald(f"Data successfully written to {file_path}")



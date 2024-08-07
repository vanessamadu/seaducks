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
        herald_count_ii = 0
        for ii in lat:
            sst_at_lat = SST_array.sel(latitude = ii,time=tt)
            SST_grad_array_x.loc[dict(time=tt, latitude=ii)] = diff1d(sst_at_lat.values, h)
            # print and log progress
            if herald_count_ii%50 == 0:
                herald(f"lat: {herald_count_ii*50/len(lat)}% complete")
                herald_count_ii +=1

        herald_count_jj = 0
        for jj in lon:
            sst_at_lon = SST_array.sel(longitude = jj,time=tt)
            SST_grad_array_y.loc[dict(time=tt, longitude=jj)] = diff1d(sst_at_lon.values, h)
            # print and log progress
            if herald_count_jj%50 == 0:
                herald(f"lat: {herald_count_jj*50/len(lon)}% complete")
                herald_count_jj +=1

        herald(f"time: {tt} complete")
    return SST_grad_array_x,SST_grad_array_y

def sst_gradient_to_da(sst,output_directory,output_filename):

    sst_gradient_x, sst_gradient_y = sst_gradient(sst)
    herald("sst gradients computed successfully")

    # rename data arrays
    sst_gradient_x.name='sst_gradient_x'
    sst_gradient_y.name='sst_gradient_y'

    # create dataset
    sst_gradient_ds = xr.Dataset(
        {
            'sst_gradient_x':sst_gradient_x,
            'sst_gradient_y':sst_gradient_y
        }
    )
    herald("dataset created successfully")
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
    
    herald("new attributes assigned successfully")
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Write to NetCDF file
    file_path = os.path.join(output_directory, output_filename)
    sst_gradient_ds.to_netcdf(file_path)

    herald(f"data successfully written to {file_path}")



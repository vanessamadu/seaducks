# seaducks/data_processing/derived_quantities.py
from seaducks import diff1d,herald,haversine_distance,inverse_distance_interpolation,format_coordinates
import xarray as xr
import os
import glob
from datetime import datetime
import numpy as np

# ----------------- SST gradient (ad hoc) ------------------ #
def sst_gradient_pointwise(sst_array: xr.DataArray, coord_str: str, time_val: np.datetime64) -> tuple:
    '''
    Calculates the Sea Surface Temperature (SST) spatial gradient in the x and y directions at a
    point coord_str.

    Parameters
    ----------
    sst_array: xr.DataArray
        SST data
    coord_str: str
        Coordinate (array-like) at which to calculate the SST gradient as a string e.g., "(50,-80)"
    time_val: np.datetime64
        Datetime at which the sst_gradient is being calculated

    Returns
    -------
    tuple
        SST gradient in the x and y directions

    Originality
    -----------
    completely original
    '''
    
    # metadata
    grid_space = 0.05 # degrees
    earth_radius = 6371 # km
    

    lat_val_str,lon_val_str = coord_str

    # find sst values near coord
    lat_neighbours = [format_coordinates(float(lat_val_str)+ii*grid_space) for ii in np.arange(-1,2,1)]
    lon_neighbours = [format_coordinates(float(lon_val_str)+jj*grid_space) for jj in np.arange(-1,2,1)]
    sst_x_neighbours = [float(sst_array.sel(time=time_val,latitude=lat_val_str, longitude=lon_val).values) if -83< float(lon_val)<-40 else np.nan for lon_val in lon_neighbours]
    sst_y_neighbours = [float(sst_array.sel(time=time_val,latitude=lat_val, longitude=lon_val_str).values)if 0 < float(lat_val) < 60 else np.nan for lat_val in lat_neighbours]
    
    #convert result to K/km
    h_lat = haversine_distance(float(lat_neighbours[0]),float(lon_val_str),
                               float(lat_neighbours[1]),float(lon_val_str))
    h_lon = haversine_distance(float(lat_val_str),float(lon_neighbours[0]),
                               float(lat_val_str),float(lon_neighbours[1]))

    return (diff1d(sst_x_neighbours,h_lon)[2],diff1d(sst_y_neighbours,h_lat)[2]) # return centre values

def interpolate_sst_gradient(drifter_lat: float, drifter_lon: float, time_val:np.datetime64, sst_array:xr.DataArray,corners:np.ndarray) -> tuple:
    '''
    Calculates the inverse distance weighted interpolated value of SST gradient at the drifter position.

    Parameters
    ----------
    drifter_lat: float 
        Latitude of drifter location being interpolated to.
    drifter_lon: float
        Longitude of drifter location being interpolated to.
    time_val: np.datetime64
        Datetime at which the sst_gradient is being calculated
    sst_array: xr.DataArray
        SST data
    corners: np.ndarray
        An array of corners (tuples) identifying grid square that the drifter location is found in.
    
    Returns
    -------
    tuple
        inverse distance weighted interpolated value of SST gradient in the x and y directions at the 
        drifter position.

    Originality
    -------
    completely original
    '''

    sst_x_gradients = []
    sst_y_gradients = []
    haversine_distances = []

    for lat_val,lon_val in corners:

        hav_distance = haversine_distance(drifter_lat,drifter_lon,lat_val,lon_val)
        haversine_distances.append(hav_distance)

        sst_x_derivative, sst_y_derivative = sst_gradient_pointwise(sst_array, (format_coordinates(lat_val),format_coordinates(lon_val)), time_val)
        sst_x_gradients.append(sst_x_derivative)
        sst_y_gradients.append(sst_y_derivative)

    return inverse_distance_interpolation(haversine_distances,sst_x_gradients), inverse_distance_interpolation(haversine_distances,sst_y_gradients)

# ----------------- SST gradient product ------------------- #

def sst_gradient(SST_array: xr.DataArray) -> tuple:
    
    # metadata
    h = 0.05                       # degrees
    earth_radius = 6371            # km
    h = np.deg2rad(h)*earth_radius # convert to km

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




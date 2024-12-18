# seaducks/data_processing/derived_quantities.py
from seaducks import diff1d,haversine_distance,inverse_distance_interpolation,format_coordinates
import xarray as xr
import numpy as np

# ----------------- SST gradient (ad hoc) ------------------ #
def sst_gradient_pointwise(sst_array: xr.DataArray, coord_str: tuple, time_val: np.datetime64) -> tuple:
    '''
    Calculates the Sea Surface Temperature (SST) spatial gradient in the x and y directions at a
    point coord_str.

    Parameters
    ----------
    sst_array: xr.DataArray
        SST data
    coord_str: tuple
        Coordinate at which to calculate the SST gradient with each value as a string e.g., ("50","-80")
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

    return (diff1d(sst_x_neighbours,h_lon)[1],diff1d(sst_y_neighbours,h_lat)[1]) # return centre values

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



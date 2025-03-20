
from seaducks import haversine_distance,inverse_distance_interpolation,format_coordinates
import numpy as np
import xarray as xr

def interpolate_new_variable(drifter_lat: float, drifter_lon: float, time_val:np.datetime64, new_var_array:xr.DataArray
                             ,corners:np.ndarray) -> tuple:
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
    new_var_array: xr.DataArray
        New variable array
    corners: np.ndarray
        An array of corners (tuples) identifying grid square that the drifter location is found in.
    
    Returns
    -------
    tuple
        inverse distance weighted interpolated value of new variable at the 
        drifter position.

    '''

    haversine_distances = []
    corner_values = []

    for lat_val,lon_val in corners:

        lat_val_str,lon_val_str = format_coordinates(lat_val), format_coordinates(lon_val)
        hav_distance = haversine_distance(drifter_lat,drifter_lon,lat_val,lon_val)

        # add new values
        haversine_distances.append(hav_distance)
        corner_values.append(float(new_var_array.sel(time=time_val,latitude=lat_val_str, longitude=lon_val_str,method='nearest').values))

    return inverse_distance_interpolation(haversine_distances,corner_values)



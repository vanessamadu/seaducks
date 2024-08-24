# seaducks/utils.py
## general
import logging
import pandas as pd
import numpy as np

## spatial filtering
import geopandas as gpd
from shapely.geometry import Polygon

# numerical differentiation
from scipy.ndimage import convolve1d
from astropy.convolution import convolve

# ------------- admin ------------ #

def herald(msg:str):
    '''
    Prints a message to the console and adds the message to the log.

    Parameters
    ----------
    msg: str
        Message to be printed and logged

    Originality
    -----------
    completely original 
    '''
    logging.info(msg)
    print(msg)

def assign_each_position_a_bin(df:pd.DataFrame, lat_grid:np.ndarray, lon_grid:np.ndarray, bin_size:float) -> pd.DataFrame:
    '''
    Assigns each drifter (lat,lon) position to a spatial bin based on latitudinal and longitudinal grids
    and a bin size.

    Parameters
    ----------
    df: pd.DataFrame
        The input containing 'lat' and 'lon' columns.
    lat_grid: np.ndarray
        The latitudes of grid points.
    lon_grid: np.ndarray
        The longitude of grid points.
    bin_size: float
        Size of the bins. Here this is only used for naming.
    
    Returns
    -------
    pd.DataFrame
        The dataframe df with columns identifying the latitudinal and longitudinal cuts
        that the drifter location is found in.

    Originality
    -----------
    adapatation with significant changes from:
        probdrift.spatialfns.lon_lat_cutter
    '''

    df.loc[:,f"lon_bin_size_{bin_size}"] = pd.cut(df["lon"], lon_grid)
    df.loc[:,f"lat_bin_size_{bin_size}"] = pd.cut(df["lat"], lat_grid)

    return df

def haversine(theta:float) -> float:
    '''
    Calculates the haversine of theta where theta is in radians.

    Parameters
    ----------
    theta: float
        Angle in radians

    Returns
    -------
    float
        Haversine(theta)

    Originality
    -----------
    completely original
    '''
    return np.sin(theta/2)**2

def haversine_distance(lat1:float, lon1:float, lat2:float, lon2:float) -> float:
    '''
    Calulcates the haversine distance (distance on a great circle) between two points on the Earth.

    Parameters
    ----------
    lat1, lat2: floats
        Latitudes of the two points in degrees.
    lon1, lon2: floats
        Longitudes of the two points in degrees.

    Returns
    -------
    float
        Haversine distance over the surface of the Earth between the two points in kilometres.

    Originality
    -----------
    completely original
    '''

    earth_radius = 6371 # km
    
    # convert latitudes and longitudes to radians
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)

    haversine_of_central_angle = haversine(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*haversine(lon2-lon1)

    return 2*earth_radius*np.arcsin(np.sqrt(haversine_of_central_angle))

def format_coordinates(coord:float) -> str:
    '''
    Convert a numerical coordinate value to a string value for use as keys.

    Parameters
    ----------
    coord: float
        Latitude or Longitude numerical value
    
    Returns
    -------
    str
        Coordinate value as a string.

    Originality
    -----------
    completely original
    '''

    if np.abs(coord) >= 10:
        string_val = format(coord,'.3f')
    elif np.abs(coord) >= 1:
        string_val = format(coord,'.3f')
    else: 
        string_val = format(coord,'.4f')

    return ''.join(string_val)


# ------------- temporal processing --------------- #

def identify_time_series_segments(timevec:pd.Series,cut_off: int = 6) -> np.ndarray:
    """
    Segments a time series based on time gaps greater than the specified cutoff.

    Parameters:
    - timevec: A pandas Series of datetime values.
    - cut_off: An integer specifying the time gap cutoff in hours.

    Returns:
    - A numpy array with segment IDs for each time point in the input series.
    
    Originality
    -----------
    as provided (up to renaming & documentation) from:
        drop_and_filter.id_aug
    """
    time_gaps = np.diff(timevec)
    mask = time_gaps > np.timedelta64(cut_off,'h') # identify where time gap is > than 'cut_off' hours
    mask = np.insert(mask,0,False)
    segments = np.cumsum(mask)
    return segments

def downsample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsamples the data to a daily temporal resolution by only including rows 
    where the hour of 'time' column is zero (midnight).

    Parameters:
    - df: A pandas DataFrame containing the data to be downsampled.

    Returns:
    - A pandas DataFrame with the downsampled data.
    
    Originality
    -----------
    as provided (up to renaming & documentation) from:  
        make_datasets.filter_dpoints 
    """
    midnight_mask = df["time"].dt.hour == 0
    return df[midnight_mask]

# --------------- spatial processing ----------------- #

def iho_region_geometry(iho_file_path: str,iho_region: str) -> Polygon:
    """
    Returns the convex hull of the geometry for a specified IHO region.

    Parameters:
    - iho_file_path: The file path to the IHO shapefile.
    - iho_region: The name of the IHO region to retrieve.

    Returns:
    - The convex hull of the specified IHO region's geometry if found.
    - None if the region is not found or an error occurs.

    Originality
    -----------
    adaptation with significant changes from:
        make_datasets.filter_dpoints & make_datasets script
    """

    world_seas = gpd.read_file(iho_file_path)
    try:
        # Set the index to "NAME" and locate the specified region
        region = world_seas.set_index("NAME").loc[iho_region]
        return region["geometry"].convex_hull
    except KeyError:
        logging.error(f'{iho_region} is not a valid IHO region')
        print(f'{iho_region} is not a valid IHO region')
        return None
    
def discard_undersampled_regions(df: pd.DataFrame, bin_size: float = 1, min_observations: int = 25):
    """
    Discards undersampled regions from a DataFrame based on spatial binning.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'lon'and 'lat' columns.
    bin_size : float, optional
        The size of the bins for longitude and latitude in degrees. Default is 1 degree.
    min_observations : int, optional
        The minimum number of observations required for a bin to be considered 
        sufficiently sampled. Default is 25.

    Returns
    -------
    pd.DataFrame
        A DataFrame with undersampled regions discarded, containing only rows from bins 
        with at least the specified minimum number of observations.


    
    """
    # set up grid

    lat_grid = np.arange(-90,90 + bin_size,bin_size)
    lon_grid = np.arange(-180,180 + bin_size,bin_size)

    df = assign_each_position_a_bin(df,lat_grid, lon_grid, bin_size=bin_size)

    bin_counts = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, observed=False).size()
    bin_counts.name = 'bin_counts'
    # assign the number of drifters in the grid box to drifters in that box

    df = df.join(bin_counts, on = [f"lon_bin_size_{bin_size}",f"lat_bin_size_{bin_size}"])
    sufficiently_sampled_mask = df['bin_counts'] > min_observations
    
    return df[sufficiently_sampled_mask]

# -------------- other processing --------------- #

def discard_undrogued_drifters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discards data associated with undrogued drifters

    Parameters:
    - df: A pandas DataFrame.

    Returns:
    - A pandas DataFrame including only drogued drifters.

    Originality
    -----------
    completely original
    """
    drogue_mask = df["drogue"].values
    return df[drogue_mask]

# --------------------- numerical differentiation -------------------------- #
    
def diff1d(row:np.ndarray,h:float) -> np.ndarray:
    ''' 
    One dimensional numerical differentiation at each point in a row/column of a grid with spacing h

    Parameters:
    -----------
    - row: A numpy array
        The input row/column from the gridded data
    - h: A float
        The grid spacing.

    Returns:
    --------
    - A numpy array of derivatives for each point in the row/column where they exist.

    Originality
    -----------
    completely original
    '''
    # define kernels
    kernel_stencil = np.array([1/(12*h),-8/(12*h),0, 8/(12*h),-1/(12*h)])[::-1]
    kernel_central = np.array([-1/(2*h),0,1/(2*h)])[::-1]
    kernel_onesided = np.array([-1/h,1/h])[::-1]
    # evaluate derivatives
    dx_stencil = stencil_mask(row,len(kernel_stencil))*convolve(row,kernel_stencil,normalize_kernel=False,nan_treatment='fill',boundary=None,
                        mask=np.isnan(row))
    dx_central = stencil_mask(row,len(kernel_central))*convolve(row,kernel_central,normalize_kernel=False,nan_treatment='fill',boundary=None,
                        mask=np.isnan(row))
    dx_right = convolve1d(row,kernel_onesided,mode="constant",cval=np.nan)
    dx_left = np.roll(dx_right,shift=1,axis=0)
    
    #decide which derivative value is used
    dx = np.where(np.isnan(dx_stencil),dx_central,dx_stencil)
    dx = np.where(np.isnan(dx),dx_left,dx)
    dx = np.where(np.isnan(dx),dx_right,dx)

    return(dx)

def stencil_mask(row: np.ndarray,kernel_len:int) -> np.ndarray:
    '''
    Determines where the 1d spatial derivative can be calculated at each grid point in the row/column
    
    Parameters
    -----------
    - row: A numpy array
        The input row/column from the gridded data
    - kernel_len: int
        The length of the convolution kernel that defines the finite difference method 
        being used for numerical differentiation

    Returns
    --------
    - A numpy array that is 1 where the stencil can be applied and NaN where it cannot.

    Originality
    -----------
    completely original
    '''
    # initialisation
    nans = np.isnan(row)            # indicator of nans in the row
    mask = np.ones(len(row)) * np.nan 
    num_neighbours = int(np.floor(kernel_len/2))

    for ii in range(num_neighbours,len(nans)-num_neighbours):
        indices = [val for val in range(ii-num_neighbours,ii+num_neighbours+1)]
        indices.pop(num_neighbours) # remove the middle value 
        # Stencil will not be able to calculate a numerical derivative if there are any NaNs in the remaining array
        if np.all(~nans[indices]): 
            mask[ii] = 1 # 1 if the stencil produces a derivative, NaN if it doesn't
    return mask

# ---------------------- interpolation --------------------- #

def get_corners(lat_cut: tuple, lon_cut: tuple) -> np.ndarray:
    '''
    Given a longitudinal and a latitudinal interval, returns the corners of the square
    spanned by the two intervals.

    Parameters
    ----------
    lat_cut: tuple
        Output of a pd.cut (latitude interval)
    lon_cut: tuple
        Output of a pd.cut (longitude interval)

    Returns
    -------
    np.ndarray
        Returns an array of coordinates of the corners of the square.

    Originality
    -----------
    adaptation with significant changes from:
        probdrift.spatialfns.gpd_functions.cuts2poly
    '''
    
    if np.array([type(lat_cut) is not float,type(lon_cut) is not float]).all():
        lon1, lon2 = lon_cut.left, lon_cut.right
        lat1, lat2 = lat_cut.left, lat_cut.right
        return np.array([(lat1, lon1), (lat1, lon2), (lat2, lon2), (lat2, lon1)])
    else:
        nan_tuple = (np.nan,np.nan)
        return np.array([nan_tuple for ii in range(4)])

def add_grid_box_corners_to_df(drifter_df: pd.DataFrame, lat_grid: np.ndarray, lon_grid: np.ndarray, bin_size=0.05):
    '''
    Originality
    ----------
    completely original
    '''

    drifter_df = assign_each_position_a_bin(drifter_df,lat_grid,lon_grid,bin_size = bin_size)
    corners = drifter_df.apply(lambda x:get_corners(x[f"lat_bin_size_{bin_size}"],x[f"lon_bin_size_{bin_size}"]),axis=1)
    drifter_df.loc[:,'corners'] = corners

    return drifter_df

def inverse_distance_interpolation(distances: np.ndarray, gridded_product_values: np.ndarray) -> float:
    '''
    Originality
    -----------
    completely original with generative AI assisted debugging (OpenAI ChatGPT).
    '''

    if np.isclose(distances,0).any():
        zero_indices = np.where(np.isclose(distances, 0))[0]
        if len(zero_indices) > 0:
            return gridded_product_values[zero_indices[0]]
        else:
            raise ValueError("No distance close to zero found.")
    else:
        inverse_distances = np.array([1/val for val in distances])

        inverse_distances = np.array([w for w,g in zip(inverse_distances,gridded_product_values) if not np.isnan(w*g)])
        weighted_gridded_product_values = np.array([w*g for w,g in zip(inverse_distances,gridded_product_values) if not np.isnan(w*g)])
        return np.sum(weighted_gridded_product_values)/np.sum(inverse_distances)


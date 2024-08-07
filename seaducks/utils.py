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
    logging.info(msg)
    print(msg)

# ------------- temporal processing --------------- #

def identify_time_series_segments(timevec:pd.Series,cut_off: int = 6) -> np.ndarray:
    """
    Segments a time series based on time gaps greater than the specified cutoff.

    Parameters:
    - timevec: A pandas Series of datetime values.
    - cut_off: An integer specifying the time gap cutoff in hours.

    Returns:
    - A numpy array with segment IDs for each time point in the input series.
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
    
def discard_undersampled_regions(df : pd.DataFrame, bin_size: float = 1, min_observations: int = 25):
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
       
    df = df.copy()
    # set up grid
    lat_grid = np.arange(-90,90 + bin_size,bin_size)
    lon_grid = np.arange(-180,180 + bin_size,bin_size)

    df.loc[:,"lon_bin"] = pd.cut(df["lon"], lon_grid)
    df.loc[:,"lat_bin"] = pd.cut(df["lat"], lat_grid)

    bin_counts = df.groupby(["lon_bin", "lat_bin"], sort=False, observed=False).size()
    bin_counts.name = 'bin_counts'
    # assign the number of drifters in the grid box to drifters in that box
    df = df.join(bin_counts, on = ["lon_bin","lat_bin"])
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
    
    Parameters:
    -----------
    - row: A numpy array
        The input row/column from the gridded data
    - kernel_len: int
        The length of the convolution kernel that defines the finite difference method 
        being used for numerical differentiation

    Returns:
    --------
    - A numpy array that is 1 where the stencil can be applied and NaN where it cannot.
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
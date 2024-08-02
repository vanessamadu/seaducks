# seaducks/utils.py
## general
import logging
import pandas as pd
import numpy as np

## spatial filtering
import geopandas as gpd
from shapely.geometry import Point, Polygon

# ------------- admin ------------ #

def herald(msg):
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




    

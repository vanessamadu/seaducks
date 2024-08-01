# seaducks/utils.py
import pandas as pd
import numpy as np

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
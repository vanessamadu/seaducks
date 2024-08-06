# seaducks/filtering.py

import numpy as np
import pandas as pd
from scipy import signal

from shapely import points,Polygon
from seaducks.utils import iho_region_geometry

# ------------- Temporal Filtering ------------ #
def butterworth_filter(time_series: np.ndarray, latitude: np.ndarray, order: int=5) -> np.ndarray: 
    """
    Applies a 1D Butterworth filter to each column of the input time series data.

    Parameters:
    - time_series: A 2D numpy array of shape (N, P) where N is the number of time points and P is 
        the number of variables.
    - latitude: A 1D numpy array of latitude values corresponding to each time point.
    - order: An integer specifying the order of the Butterworth filter. Default value is 5.

    Returns:
    - A 2D numpy array of the same shape as the input array, with filtered data.
    """
    time_series_len,num_time_series = time_series.shape

    # initialise output with same shape and dtype as input
    out = np.zeros(time_series.shape,dtype=time_series.dtype) 

    # temporarily set missing values to zero
    nan_mask = np.isnan(time_series)

    # prevent changes to the time series outside of this function
    time_series = time_series.copy()
    time_series[nan_mask] = 0

    sample_freq = 1/(6*60*60) #Hz
    nyquist_freq = 0.5*sample_freq 

    if time_series_len < order * 6:
        out = out*np.nan # discard time series segment if too short
    else:
        # perform daily filtering (moving BW filter over four six hourly observations)
        for ii in range(0,time_series_len,4):
            average_24_hour_lat = np.mean(latitude[ii:(ii+4)])

            # threshold frequency = minimum of intertial frequency/1.5 and once every five days
            earth_rotation_rate = 7.2921e-5
            inertial_freq = 2*earth_rotation_rate*np.sin(np.deg2rad(np.abs(average_24_hour_lat)))
            inertial_freq_hz = inertial_freq/(2*np.pi)
            five_day_freq_hz = 1 / (5 * 24 * 60 ** 2)
            threshold_freq = np.max([inertial_freq_hz/1.5, five_day_freq_hz]) 

            b,a = signal.butter(order,threshold_freq/nyquist_freq,btype='lowpass',analog=False)

            for jj in range(num_time_series):
                filtered_time_series = signal.filtfilt(b,a,time_series[:,jj])
                out[ii:(ii+4),jj] = filtered_time_series[ii:(ii+4)]
        out[nan_mask] = np.nan
    return out

def apply_butterworth_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the butterworth_filter function to each covariate column in the DataFrame.

    This function filters the 'u', 'v', 'Wx', 'Wy', 'Tx', and 'Ty' columns of the DataFrame,
    creating copies if they do not already exist, and replacing the filtered values back
    into the DataFrame.

    Parameters:
    - df: A pandas DataFrame containing the data to be filtered. The DataFrame must contain
          a 'lat' column and the columns specified in cols_to_filter.

    Returns:
    - A pandas DataFrame with the filtered data.
    """
    lat = df['lat'].values
    time_dependent_vars = ['u','v','Wx','Wy','Tx','Ty']

    # prevent changes to the data outside of this function
    for var in time_dependent_vars:
        if var + '_filtered' not in df.columns:
            df[var + '_filtered'] = df[var].copy()
    vars_to_filter = [var + '_filtered' for var in time_dependent_vars]
    
    time_series = df[vars_to_filter].values
    filtered_vars = butterworth_filter(time_series,lat)

    df[vars_to_filter] = filtered_vars
    return df


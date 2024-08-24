# scripts/sst_interpolation.py
import sys
import os
import pandas as pd
import xarray as xr
import numpy as np
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seaducks.data_processing.derived_quantities import interpolate_sst_gradient
from seaducks.utils import add_grid_box_corners_to_df,format_coordinates

# Configure logging
logging.basicConfig(filename=os.path.join('logs', 'sst_interpolation.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():

    filename =  'corrected_velocity_drifter_full.h5'
    output_path = 'corrected_velocity_drifter_with_sst_gradient.h5'


    sst_array_path = "Met_Office_West_NA_sst_1993_2019_full.nc"
    sst_dataset = xr.open_dataset(sst_array_path)
    sst_array = sst_dataset['analysed_sst']

    # ensure correct look up (remove issues with machine precision)
    lat_str = sst_array.coords['latitude'].values.astype(str)  # Convert lat values to strings
    lon_str = sst_array.coords['longitude'].values.astype(str)  # Convert lat values to strings
    # assign string coordinates
    sst_array = sst_array.assign_coords(lat_str=('latitude', lat_str))
    sst_array = sst_array.assign_coords(lon_str=('longitude', lon_str))

    dataset = pd.read_hdf(filename)
    lon_lim_W = -83
    lon_lim_E = -40
    lat_lim_S = 0
    lat_lim_N = 60

    df = dataset.query('@lon_lim_W < lon < @lon_lim_E').copy()
    df = df.query('@lat_lim_S < lat < @lat_lim_N').head(50).copy()

    # add corners
    lat_grid = np.array(sst_array['latitude'])
    lon_grid = np.array(sst_array['longitude'])
    bin_size = 0.05

    drifter_df = add_grid_box_corners_to_df(df, lat_grid, lon_grid, bin_size=bin_size)

    # sst is daily
    
    condition = drifter_df['time'].dt.time == pd.Timestamp('00:00:00').time()

    sst_gradient = drifter_df[condition].apply(lambda row: interpolate_sst_gradient(format_coordinates(row['lat']),format_coordinates(row['lon']),row['time'],sst_array,row['corners']),
    axis=1,
    result_type='expand')

    drifter_df.loc[condition,'sst_x_derivative'] = sst_gradient.iloc[:,0]
    drifter_df.loc[condition,'sst_y_derivative'] = sst_gradient.iloc[:,1]

    drifter_df = drifter_df.drop(['corners',f"lat_bin_size_{bin_size}",f"lon_bin_size_{bin_size}"],axis =1)
    drifter_df.to_hdf(path_or_buf=output_path, key="drifter", mode='w',format="table")


if __name__ == '__main__':
    main()
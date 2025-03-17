# scripts/run_interpolate_new_variable

import sys
import os
import pandas as pd
import xarray as xr
import numpy as np
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seaducks.data_processing import interpolate_new_variable
from seaducks import add_grid_box_corners_to_df

################# INFORMATION THAT NEEDS TO BE ADJUSTED ##########################
new_variable_name = ''
bin_size = 0.05

# spatial extent
lon_lim_W = -83
lon_lim_E = -40
lat_lim_S = 0
lat_lim_N = 60

# temporal resolution
data_sample_times = ["00:00:00"]

# file paths
filename =  'corrected_velocity_drifter_full.h5'
output_path = f'corrected_velocity_drifter_with_{new_variable_name}.h5'
new_variable_data_path = ""
##################################################################################

# Configure logging
logging.basicConfig(filename=os.path.join('logs', f'{new_variable_name}_interpolation.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():

    # new variable data
    new_variable_dataset = xr.open_dataset(new_variable_data_path)
    new_variable_array = new_variable_dataset['analysed_sst']

    # ensure correct look up (remove issues with machine precision)
    lat_str = new_variable_array.coords['latitude'].values.astype(str)  # Convert lat values to strings
    lon_str = new_variable_array.coords['longitude'].values.astype(str)  # Convert lat values to strings
    # assign string coordinates
    new_variable_array = new_variable_array.assign_coords(lat_str=('latitude', lat_str))
    new_variable_array = new_variable_array.assign_coords(lon_str=('longitude', lon_str))

    dataset = pd.read_hdf(filename)

    df = dataset.query('@lon_lim_W < lon < @lon_lim_E').copy()
    df = df.query('@lat_lim_S < lat < @lat_lim_N').copy()

    # add corners
    lat_grid = np.array(new_variable_array['latitude'])
    lon_grid = np.array(new_variable_array['longitude'])

    drifter_df = add_grid_box_corners_to_df(df, lat_grid, lon_grid, bin_size=bin_size)

    # set condition to match the temporal resolution of the new variable data
    
    condition = drifter_df['time'].dt.time in [pd.Timestamp(timestamp).time() for timestamp in data_sample_times]

    new_variable = drifter_df[condition].apply(lambda row: interpolate_new_variable(row['lat'],row['lon'],row['time'],new_variable_array,row['corners']),
    axis=1,
    result_type='expand')

    drifter_df.loc[condition,new_variable_name] = new_variable

    drifter_df = drifter_df.drop(['corners',f"lat_bin_size_{bin_size}",f"lon_bin_size_{bin_size}"],axis =1)
    drifter_df.to_hdf(path_or_buf=output_path, key="drifter", mode='w',format="table")


if __name__ == '__main__':
    main()
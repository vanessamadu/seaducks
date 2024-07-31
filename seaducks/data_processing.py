# seaducks/data_processing.py
import pandas as pd
import numpy as np
import logging
from seaducks.filtering import filter_covariates
from seaducks.utils import identify_time_series_segments
from seaducks.config import config
import time

def time_series_processing(file_path: str, output_path: str, sample_proportion: float = 1):
    overall_start_time = time.time()

    try:
        dataset = pd.read_hdf(file_path)
        logging.info('dataset loaded')

        # sub-sample
        if sample_proportion < 1:
            dataset = dataset.sample(frac=sample_proportion, random_state=42)
            logging.info("dataset sampled successfully")
        print(f'{sample_proportion*100}% of the dataset | new dataset shape: {dataset.shape}')

        # convert from cm/s -> m/s 
        dataset.loc[:, 'u']/=100**2
        dataset.loc[:, 'v']/=100**2

        # set extreme values to NaN
        for var in ['u','v','Tx','Ty','Wy','Wx']:
            extreme_val_mask = dataset[var] < -900
            dataset.loc[extreme_val_mask,var] = np.nan

        # discard observations with lat/lon variance estimate >= 0.5 degrees
        dataset = dataset.query('lon_var<0.5 and lat_var<0.5').copy()

        # group the data for each drifter id into time series segments 
        dataset.loc[:,'segment_id'] = dataset.groupby('id')['time'].transform(identify_time_series_segments)

        filtered_dataset = dataset.groupby(['id', 'segment_id'], group_keys=False).apply(filter_covariates, include_groups=False)
        print('applied Butterworth filter to each segment')

        filtered_dataset.to_hdf(path_or_buf=output_path, key="drifter", mode='w')
        print('saved filtered data')

        elapsed_time = time.time() - overall_start_time
        print(f"Filtering {sample_proportion*100}% of the data took : {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f'an error has occured: {e}')
        print(f'an error has occured: {e}')
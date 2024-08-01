# seaducks/data_processing.py
import pandas as pd
import numpy as np
import logging
from seaducks.filtering import temporal_filter_covariates
from seaducks.utils import identify_time_series_segments,downsample_to_daily
from seaducks.config import config
import time

def time_series_processing(file_path: str, output_path: str, sample_proportion: float = 1,seed=config['random_state']):
    """
    Processes a dataset of drifter data, applying various data cleaning and filtering steps,
    and saves the processed dataset to the specified output path.

    Parameters:
    -----------
    file_path : str
        The path to the input HDF5 file containing the raw drifter data.
    output_path : str
        The path where the processed HDF5 file will be saved.
    sample_proportion : float, optional, default=1
        The proportion of the dataset to sample. If less than 1, a random sample of the specified 
        proportion is taken from the dataset.
    seed : int, optional, default=config['random_state']
        The random seed for reproducibility during sampling.

    Processing Steps:
    -----------------
    1. Load the dataset from the specified HDF5 file.
    2. Optionally subsample the dataset based on `sample_proportion`.
    3. Convert velocity units from cm/s to m/s for 'u' and 'v' columns.
    4. Set extreme values (less than -900) to NaN for columns 'u', 'v', 'Tx', 'Ty', 'Wy', 'Wx'.
    5. Discard observations with lat/lon variance estimates greater than or equal to 0.5 degrees.
    6. Group the data into 6-hourly segments based on the 'id' and 'time' columns.
    7. Apply a Butterworth filter to each segment.
    8. Downsample the dataset to daily resolution (include only midnight observations)
    9. Save the processed dataset to the specified output path in HDF5 format.

    Logging:
    --------
    Logs messages at various stages of processing to track progress and capture any errors.

    Exception Handling:
    -------------------
    Catches and logs any exceptions that occur during processing, and prints an error message.

    Examples:
    ---------
    To process the entire dataset:
    >>> time_series_processing('data/drifter_full.h5', 'data/processed_data.h5')

    To process a 10% sample of the dataset:
    >>> time_series_processing('data/drifter_full.h5', 'data/processed_sampled_data.h5', sample_proportion=0.1)
    """
    overall_start_time = time.time()


    try:
        raw_dataset = pd.read_hdf(file_path)
        dataset = raw_dataset.copy() # make sure dataset is not impacted outside of the function
        logging.info('dataset loaded')

        # sub-sample
        if sample_proportion < 1:
            dataset = dataset.sample(frac=sample_proportion, random_state=seed)
            logging.info("dataset sampled successfully")
        print(f'{sample_proportion*100}% of the dataset | new dataset shape: {dataset.shape}')

        # convert from cm/s -> m/s 
        dataset.loc[:, 'u']/=100
        dataset.loc[:, 'v']/=100
        logging.info("drifter velocity converted to m/s successfully")
        print(f'drifter velocity converted to m/s successfully')

        # set extreme values to NaN
        for var in ['u','v','Tx','Ty','Wy','Wx']:
            extreme_val_mask = dataset[var] < -900
            dataset.loc[extreme_val_mask,var] = np.nan
        print(f'extreme values set to nan')

        # group the data for each drifter id into time series segments 
        dataset.loc[:,'segment_id'] = dataset.groupby('id')['time'].transform(identify_time_series_segments)
        print(f'6-hourly segments identified')
        filtered_dataset = dataset.groupby(['id', 'segment_id'], group_keys=False).apply(temporal_filter_covariates, include_groups=False)
        print('applied Butterworth filter to each segment')

        # downsample to daily resolution
        filtered_dataset = downsample_to_daily(filtered_dataset)
        print('filtered data downsampled to daily')

        filtered_dataset.to_hdf(path_or_buf=output_path, key="drifter", mode='w')
        print('saved filtered data')

        elapsed_time = time.time() - overall_start_time
        print(f"Filtering {sample_proportion*100}% of the data took : {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f'an error has occured: {e}')
        print(f'an error has occured: {e}')
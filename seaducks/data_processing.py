# seaducks/data_processing.py
import pandas as pd
import numpy as np
from shapely import Polygon,points
from seaducks.filtering import apply_butterworth_filter
from seaducks.utils import downsample_to_daily,herald,discard_undrogued_drifters,identify_time_series_segments
from seaducks.utils import discard_undersampled_regions
from seaducks.config import config
import time
'''

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
        print(f'an error has occured: {e}')'''
    
def data_filtering(region: Polygon,file_path: str, output_path: str, sample_proportion: float = 1,seed=config['random_state'],
                   lon_lim_W: float =-83, lon_lim_E: float = 40):
    
    overall_start_time = time.time()

    try:
    
        # initialisation
        raw_dataset = pd.read_hdf(file_path)
        df = raw_dataset.copy()
        herald('data loaded successfully')

        ## sample dataset (optional)
        if sample_proportion < 1:
            df = df.sample(frac=sample_proportion, random_state=seed)
            herald('data sampled successfully')
        herald(f'{sample_proportion*100}% of the dataset | new dataset shape: {df.shape}')

        # 0) data-preprocess: clean and convert
        ## convert from cm/s -> m/s 
        df.loc[:, 'u']/=100
        df.loc[:, 'v']/=100
        herald("drifter velocity converted to m/s successfully")

        ## set extreme values to NaN
        for var in ['u','v','Tx','Ty','Wy','Wx']:
            extreme_val_mask = df[var] < -900
            df.loc[extreme_val_mask,var] = np.nan
        herald(f'extreme values set to nan')

        # 1) discard undrogued drifters
        df = discard_undrogued_drifters(df).copy()
        herald('undrogued drifters discarded successfully')

        # 2) discard observations with a lat or lon variance estimate > 0.25 degrees
        df = df.query('lon_var<0.25 and lat_var<0.25').copy()
        herald('observations with variance lat/lon estimates more than 0.25 degrees discarded')

        #3) discard data that is not in the North Atlantic and in the region [83W, 40W]
        # remove data outside of the interval [-83,40]
        df = df.query('@lon_lim_W < lon < @lon_lim_E').copy()
        drifter_locs = points(df[["lon","lat"]].values).tolist() # (lon,lat) in (x,y) form for geometry
        region_mask = [loc.within(region) for loc in drifter_locs]
        df = df[region_mask].copy() 

        # 4) discard data in undersampled regions
        df = discard_undersampled_regions(df).copy()

        # 5) split time series into six hourly segments for each drifter. discard segments that are 
        #    very short. Apply a fifth order Butterworth filter.
            
        ## group the data for each drifter id into time series segments 
        df.loc[:,'segment_id'] = df.groupby('id')['time'].transform(identify_time_series_segments)
        herald('6-hourly segments identified')
        df = df.groupby(['id', 'segment_id'], group_keys=False).apply(apply_butterworth_filter, include_groups=False).copy()
        herald('applied Butterworth filter to each segment')

        # 6) downsample to daily temporal resolution
        df = downsample_to_daily(df).copy()

        df.to_hdf(path_or_buf=output_path, key="drifter", mode='w')
        herald('saved filtered data')

        elapsed_time = time.time() - overall_start_time
        herald(f"Filtering {sample_proportion*100}% of the data took : {elapsed_time:.2f} seconds")

    except Exception as e:
        herald(f'an error has occured: {e}')


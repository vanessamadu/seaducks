# seaducks/data_processing/data_processing.py
import pandas as pd
import numpy as np
from shapely import Polygon,points
from seaducks.data_processing.filtering import apply_butterworth_filter
from seaducks.utils import downsample_to_daily,herald,discard_undrogued_drifters,identify_time_series_segments
from seaducks.utils import discard_undersampled_regions
from seaducks.config import config
import time
    
def data_filtering(region: Polygon,file_path: str, output_path: str, sample_proportion: float = 1,seed=config['random_state'],
                   lon_lim_W: float =-83, lon_lim_E: float = -40):
    
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

        ## set missing values to NaN 
        ### Note: Missing wind stress and wind speed are set to -2147483647 and -32767 in the source data. respectively.
        for var in ['Tx','Ty','Wy','Wx']:
            missing_val_mask = df[var] < -900
            df.loc[missing_val_mask,var] = np.nan
        herald(f'missing values set to nan')

        # 1) discard data that is not in the North Atlantic and in the region [83W, 40W]
        # remove data outside of the interval [-83,-40]
        df = df.query('@lon_lim_W < lon < @lon_lim_E').copy()
        drifter_locs = points(df[["lon","lat"]].values).tolist() # (lon,lat) in (x,y) form for geometry
        region_mask = [loc.within(region) for loc in drifter_locs]
        df = df[region_mask].copy() 

        # 2) split time series into six hourly segments for each drifter. discard segments that are 
        #    very short. Apply a fifth order Butterworth filter.
            
        ## group the data for each drifter id into time series segments 
    
        df.loc[:,'segment_id'] = df[['id','time']].groupby('id')['time'].transform(identify_time_series_segments)
        herald('6-hourly segments identified')
        variables = list(df.columns)
        df = df.groupby(['id', 'segment_id'])[variables].apply(apply_butterworth_filter).copy()
        herald('applied Butterworth filter to each segment')

        # 3) discard undrogued drifters
        df = discard_undrogued_drifters(df).copy()
        herald('undrogued drifters discarded successfully')

        # 4) discard observations with a lat or lon variance estimate > 0.25 degrees
        df = df.query('lon_var<0.25 and lat_var<0.25').copy()
        herald('observations with variance lat/lon estimates more than 0.25 degrees discarded')

        # 5) downsample to daily temporal resolution
        df = downsample_to_daily(df).copy()
        herald('data downsampled to daily')

        # 6) discard data in undersampled regions
        df = discard_undersampled_regions(df).copy()

        # reset indices
        df = df.drop(['segment_id','id'],axis=1).reset_index().copy()

        df[config['return_variables']].to_hdf(path_or_buf=output_path, key="drifter", mode='w',format="fixed")
        herald('saved filtered data')
        

        elapsed_time = time.time() - overall_start_time
        herald(f"Filtering {sample_proportion*100}% of the data took : {elapsed_time:.2f} seconds")

    except Exception as e:
        herald(f'an error has occured: {e}')

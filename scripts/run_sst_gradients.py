import time
import os
import xarray as xr
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seaducks.data_processing.derived_quantities import sst_gradient_to_da
from seaducks.utils import herald
import logging

# Configure logging
logging.basicConfig(filename=os.path.join('logs', 'sst_gradients.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    start = time.time()

    # configure directories
    filename=r"CMEMS_West_NA_sst_2012_2022.nc"
    file_directory = r"D:\PhD\ocean-datasets\copernicus-data"
    data_path = os.path.join(file_directory, filename)

    output_directory = r"D:\PhD\ocean-datasets\derived-data\sst"
    output_filename = r"sst_gradients_t0_t3.nc"
    # Load the NetCDF file
    
    dataset = xr.open_dataset(data_path)

    sst = dataset['analysed_sst'].isel(time=list(range(3)))

    sst_gradient_to_da(sst,output_directory,output_filename)

    herald(time.time()-start)

if __name__ == '__main__':
    main()



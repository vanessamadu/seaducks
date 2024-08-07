import time
import os
import xarray as xr
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
    filename="CMEMS_West_NA_sst_2012_2022.nc"
    file_directory = "D:\PhD\ocean-datasets\copernicus-data"
    data_path = os.path.join(file_directory, filename)

    output_directory = "D:\PhD\ocean-datasets\derived_quantities\sst"
    output_filename = "sst_gradients_t0_t3.nc"
    output_path = os.path.join(output_directory, output_filename)
    # Load the NetCDF file
    
    dataset = xr.open_dataset(data_path)

    sst = dataset['analysed_sst'].isel(time=list(range(3)))

    sst_gradient_to_da(sst,output_path)

    herald(time.time()-start)

if __name__ == '__main__':
    main()
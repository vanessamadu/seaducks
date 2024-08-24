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
    filename = "Met_Office_West_NA_sst_1993_2019_full.nc"
    file_directory = r"./"

    output_directory = r"./"
    output_filename = r"Met_Office_West_NA_sst_gradients_1993_2019.nc"
    # Load the NetCDF file

    sst_gradient_to_da(file_directory,filename,output_directory,output_filename)

    herald(time.time()-start)

if __name__ == '__main__':
    main()





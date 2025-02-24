# seaducks/run_processing.py
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seaducks.data_processing.data_processing import data_filtering
from seaducks import iho_region_geometry, herald

# Configure logging
logging.basicConfig(filename=os.path.join('logs', 'data_processing.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    iho_file_path = os.path.join('data', 'world_seas_iho_v3')
    iho_region = 'North Atlantic Ocean'
    region = iho_region_geometry(iho_file_path,iho_region)
    herald(f'region: {iho_region} created successfully')
    file_path = os.path.join('data', 'corrected_velocity_drifter_set_sst_gradient.h5')
    output_path = os.path.join('data', 'filtered_nao_drifters_with_sst_gradient_005.h5')
    
    data_filtering(region, file_path, output_path)

if __name__ == '__main__':
    main()
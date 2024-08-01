# seaducks/run_processing.py
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seaducks.data_processing import time_series_processing

# Configure logging
logging.basicConfig(filename=os.path.join('logs', 'data_processing.log'), 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    file_path = os.path.join('data', 'corrected_velocity_drifter_full.h5')
    output_path = os.path.join('data', 'filtered_data_test.h5')
    
    time_series_processing(file_path, output_path,sample_proportion=0.075)

if __name__ == '__main__':
    main()
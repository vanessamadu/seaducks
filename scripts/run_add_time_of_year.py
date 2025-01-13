# scripts/correct_velocity.py
'''
description:    script to correct the drifter velocity after inherited incorrect initial processing
                only run on `drifter_full.h5`

    ->  divide drifter velocities by 100 to correct initial processing that multiplied 
        cm/s by 100 to get m/s
'''
import pandas as pd
import os

def main():

    file_path = os.path.join('data', 'drifter_full.h5')
    output_path = os.path.join('data', 'corrected_velocity_drifter_full.h5')

    dataset = pd.read_hdf(file_path)
    # data correction
    dataset.loc[:,'u']/=100
    dataset.loc[:,'v']/=100
    dataset.to_hdf(output_path, key="drifter", mode='w')

if __name__ == '__main__':
    main()
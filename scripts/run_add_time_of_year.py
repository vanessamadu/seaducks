import pandas as pd
import os

def main():

    file_path = os.path.join('data', 'filtered_nao_drifters_with_sst_gradient.h5')
    output_path = os.path.join('data', 'filtered_nao_drifters_with_sst_gradient_with_time.h5')

    dataset = pd.read_hdf(file_path)
    # data correction
    dataset['day_of_year'] = dataset['time'].apply(lambda t : t.timetuple().tm_yday)

    dataset.to_hdf(output_path, key="drifter", mode='w')

if __name__ == '__main__':
    main()
import pandas as pd
import os
import numpy as np

def main():

    file_path = os.path.join('data', 'archived_data_sets/filtered_nao_drifters_with_sst_gradient_with_time.h5')
    output_path = os.path.join('data', 'complete_filtered_nao_drifter_dataset.h5')

    dataset = pd.read_hdf(file_path)
    variable_names = [['Wx', 'Wy'], ['Tx', 'Ty'], ['u_av', 'v_av']]
    new_variable_names = [['R_wind_speed','arg_wind_speed'], ['R_wind_stress','arg_wind_stress'], ['R_vel_av','arg_vel_av']]

    for ii, variable_name in enumerate(variable_names):
        aux_vals = dataset.apply(lambda row: row[variable_name[0]] + row[variable_name[1]]*1j,axis=1)
        dataset[new_variable_names[ii][0]] = np.abs(aux_vals)
        dataset[new_variable_names[ii][1]] = np.angle(aux_vals,deg=True)
    dataset.to_hdf(output_path, key="drifter", mode='w')

if __name__ == '__main__':
    main()
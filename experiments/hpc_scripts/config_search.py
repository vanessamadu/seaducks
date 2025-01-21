import pickle
import numpy as np
 
def model_config_verbose(experiment_ID,model_filepath):
    with open(model_filepath,'rb') as pickle_file:
        model_configs = pickle.load(pickle_file)
    config_id = int(np.floor(experiment_ID/10))
    config = model_configs[config_id]

    # ------------ print useful information -------------- #
    print(f'config ID: {config_id}')
    print('\n Hyperparameter Values')
    print('---------------------------')

    print(f'eta: {config_id[0]}')
    print(f'min. data per leaf: {config_id[1]}')
    print(f'max. leaves per node: {config_id[2]}')

    print('\n Data Information')
    print('---------------------------')

    print(f'Uses SST gradient: {config_id[3]}')
    print(f'Uses velocities in polar form: {config_id[4]}')

    return config




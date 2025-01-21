import pickle
import numpy as np
 
def model_config(experiment_ID,model_filepath,*,
                         verbose=False):
    with open(model_filepath,'rb') as pickle_file:
        model_configs = pickle.load(pickle_file)
    config_id = int(np.floor(experiment_ID/10))
    config = model_configs[config_id]

    if verbose:
        # ------------ print useful information -------------- #
        print(f'\nexperiment ID: {experiment_ID}')
        print(f'config ID: {config_id}')
        print(f'replication: {(experiment_ID-1)%10}')
        print('\n Hyperparameter Values')
        print('---------------------------')

        print(f'learning rate: {config[0]}')
        print(f'min. data per leaf: {config[1]}')
        print(f'max. leaves per node: {config[2]}')

        print('\n Data Information')
        print('---------------------------')

        print(f'Uses SST gradient: {config[3]}')
        print(f'Uses velocities in polar form: {config[4]}')

    return config




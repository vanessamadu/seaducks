from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def train_test_validation_split(X, Y,*,
                                test_frac = 0.10, validation_frac = 0.09, 
                                random_seed_idx = None, shuffle = True, stratify = None):
    X_aux, X_test, Y_aux, Y_test = train_test_split(X, Y, 
                                                        test_size=test_frac, random_state = np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
    if validation_frac == 0:
        return X_aux, X_test, Y_aux, Y_test
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(X_aux, Y_aux,
                                                        test_size=validation_frac/(1 - test_frac), random_state = np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
        return X_train, X_test, X_val, Y_train, Y_test, Y_val
    
def train_test_validation_split_ids(ids,*,
                                test_frac = 0.10, validation_frac = 0.09, 
                                random_seed_idx = None, shuffle = True, stratify = None, masks=False):
    id_aux, id_test  = train_test_split(ids, 
                                        test_size=test_frac, random_state= np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
    if validation_frac == 0:
        if masks:
            return np.in1d(ids,id_aux), np.in1d(ids,id_test)
        else:
            return id_aux, id_test
    else:
        id_train, id_val = train_test_split(id_aux,
                                                        test_size=validation_frac/(1 - test_frac), random_state= np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
        if masks:
            return np.in1d(ids,id_train), np.in1d(ids,id_test), np.in1d(ids,id_val)
        else:
            return id_train, id_test, id_val
    
def nominal_cluster_sampling(data,*,
                             test_frac = 0.10, validation_frac = 0.09, 
                             tol = 5e-5, num_seeds = 1):
    jj = 0

    number_of_samples = len(data.index)
    count_by_id = data.groupby('id').size()
    X, Y = np.array(count_by_id.index), np.array(count_by_id)
    train_frac = 1-test_frac-validation_frac

    actual_test_frac = 0.0
    actual_train_frac = 0.0
    actual_validation_frac = 0.0
    
    seeds = []

    while len(seeds)<num_seeds:
            
        seed = np.random.seed(jj)
        
        _,_,_,Y_train,Y_test,Y_val = train_test_validation_split(X, Y,
                                                                    test_frac = test_frac, validation_frac = validation_frac, random_state=seed)
        actual_train_frac, actual_test_frac, actual_validation_frac = np.array([sum(Y_train),sum(Y_test),sum(Y_val)])/number_of_samples

        if (np.abs(
        np.array([actual_test_frac - test_frac
                           , actual_train_frac -train_frac, 
                           actual_validation_frac - validation_frac]) < tol).all()
                           ):
            seeds.append(jj)
            print([actual_test_frac, actual_train_frac, actual_validation_frac])

        jj += 1
    return seeds

def model_config(config_id,model_filepath,*,
                         verbose=False):
    """
    Load and return a specific model configuration hyperparameter settings from a 
    pickled file.

    Parameters:
    -----------
    config_id : int
        Index of the desired configuration in the dictionary of configurations.
    model_filepath : str
        Path to the pickled file containing stored model configurations.
    verbose : bool, optional (default=False)
        If True, prints detailed information about the selected configuration, 
        including hyperparameter values and data usage options.

    Returns:
    --------
    config : tuple
        A tuple containing the selected configuration's hyperparameters and settings:
        (learning_rate, min_data_per_leaf, max_leaves_per_node, 
         uses_SST_gradient, uses_polar_form_velocities)

    Notes:
    ------
    The expected format of the pickle file is a dictionary where each 
    item is a tuple of the form described above.
    """

    with open(model_filepath,'rb') as pickle_file:
        model_configs = pickle.load(pickle_file)
    config = model_configs[config_id]

    if verbose:
        # ------------ print useful information -------------- #
        print(f'\nconfig ID: {config_id}')

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

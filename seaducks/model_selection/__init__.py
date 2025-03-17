from sklearn.model_selection import train_test_split
import numpy as np
import pickle
# for typehinting
from pandas import DataFrame
from numpy import ndarray
from pyvista import MatrixLike, ArrayLike

def train_test_validation_split(X: ArrayLike, Y: ArrayLike,*,
                                test_frac: float = 0.10, validation_frac: float = 0.09, 
                                random_seed_idx: (int|None) = None, shuffle: bool = True, stratify: (ArrayLike|None) = None) -> MatrixLike:
    """
    Split data into training, validation, and test sets.

    Parameters:
    -----------
    X : array-like
        Explanatory variable dataset.
    Y : array-like
        Response variable dataset.
    test_frac : float, optional (default=0.10)
        Fraction of the data to allocate to the test set.
    validation_frac : float, optional (default=0.09)
        Fraction of the data to allocate to the validation set.
    random_seed_idx : int or None, optional
        Random seed index.
    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data before splitting.
    stratify : array-like or None, optional
        If not None, data is split in a stratified fashion using this array.

    Returns:
    --------
    If validation_frac == 0:
        X_aux, X_test, Y_aux, Y_test : arrays
            Training set and test set splits.
    Else:
        X_train, X_test, X_val, Y_train, Y_test, Y_val : arrays
            Full train, test, and validation splits.
    """
    X_aux, X_test, Y_aux, Y_test = train_test_split(X, Y, 
                                                        test_size=test_frac, random_state = np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
    if validation_frac == 0:
        return X_aux, X_test, Y_aux, Y_test
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(X_aux, Y_aux,
                                                        test_size=validation_frac/(1 - test_frac), random_state = np.random.seed(random_seed_idx), shuffle = shuffle, stratify = stratify)
        return X_train, X_test, X_val, Y_train, Y_test, Y_val
    
def train_test_validation_split_ids(ids: ArrayLike,*,
                                test_frac: float = 0.10, validation_frac: float = 0.09, 
                                random_seed_idx: int = None, shuffle: bool = True, stratify:(ArrayLike|None) = None, masks: bool=False) -> MatrixLike:
    """
    Split a list of IDs into train, validation, and test sets or masks.

    Parameters:
    -----------
    ids : array-like
        Array of IDs to split.
    test_frac : float, optional (default=0.10)
        Fraction of the IDs to allocate to the test set.
    validation_frac : float, optional (default=0.09)
        Fraction of the IDs to allocate to the validation set.
    random_seed_idx : int or None, optional
        Random seed index.
    shuffle : bool, optional (default=True)
        Whether or not to shuffle the IDs before splitting.
    stratify : array-like or None, optional
        If not None, IDs are split in a stratified fashion using this array.
    masks : bool, optional (default=False)
        If True, returns boolean masks instead of ID arrays.

    Returns:
    --------
    If validation_frac == 0:
        id_aux, id_test : arrays or masks
            Train and test ID splits or their corresponding boolean masks.
    Else:
        id_train, id_test, id_val : arrays or masks
            Train, test, and validation ID splits or their corresponding boolean masks.
    """
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
    
def get_nominal_cluster_sampling_seeds(data: DataFrame,*,
                             test_frac: float = 0.10, validation_frac: float = 0.09, 
                             tol: float = 5e-5, num_seeds: int = 1) -> ndarray:
    """
    Find random seeds that ensure proportions of data in the train, validation, and test sets
    match nominal values for cluster sampling.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing an 'id' column representing drifter IDs for cluster sampling.
    test_frac : float, optional (default=0.10)
        Target fraction of data to allocate to the test set.
    validation_frac : float, optional (default=0.09)
        Target fraction of data to allocate to the validation set.
    tol : float, optional (default=5e-5)
        Tolerance level for the deviation of actual split fractions from target values.
    num_seeds : int, optional (default=1)
        Number of valid random seeds to find.

    Returns:
    --------
    seeds : array (containing integers)
        List of random seeds that yield splits meeting the specified tolerance criteria.

    Notes:
    ------
    The function repeatedly samples splits until it finds `num_seeds` random seeds 
    for which the resulting splits closely match the desired fractions within `tol`.
    """
    
    # initialise
    jj = 0
    actual_test_frac = 0.0
    actual_train_frac = 0.0
    actual_validation_frac = 0.0
    train_frac = 1-test_frac-validation_frac
    seeds = []

    number_of_samples = len(data.index)
    count_by_id = data.groupby('id').size()
    X, Y = np.array(count_by_id.index), np.array(count_by_id)

    while len(seeds)<num_seeds:
            
        seed = np.random.seed(jj)
        
        _,_,_,Y_train,Y_test,Y_val = train_test_validation_split(X, Y,
                                                                    test_frac = test_frac, validation_frac = validation_frac, random_state=seed)
        actual_train_frac, actual_test_frac, actual_validation_frac = np.array([sum(Y_train),sum(Y_test),sum(Y_val)])/number_of_samples

        # save seed if it is within tolerance of the nominal split
        if (np.abs(
        np.array([actual_test_frac - test_frac
                           , actual_train_frac -train_frac, 
                           actual_validation_frac - validation_frac]) < tol).all()
                           ):
            seeds.append(jj)
            print([actual_test_frac, actual_train_frac, actual_validation_frac])

        jj += 1
    return seeds

def get_info_from_config(config_id:int,model_filepath:str,*,
                         verbose:bool=False) -> tuple:
    """
    Load and return a specific model configuration's hyperparameter settings from a 
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

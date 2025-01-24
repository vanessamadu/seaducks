import sys
import os
from datetime import datetime
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/config'), '..')))
from seaducks.config import config
import pandas as pd
from seaducks.models._mvn_ngboost import MVN_ngboost
from sklearn.tree import DecisionTreeRegressor
import time
import pickle
import numpy as np


if __name__=='__main__':

    start = time.time()
    # --------- set up --------- #
    # load configuration ID look up
    with open('./model_configuration_ids.p', 'rb') as pickle_file:
        configurations_dict = pickle.load(pickle_file)
    
    num_reps = 100
    config_ids = [] # optimal hyperparameters

    # initialise indexing
    index = int(sys.argv[1])
    rep = int(index-1)%num_reps
    config_id = config_ids[int(np.floor((index-1)/num_reps))]
    
    # initialise hyperparameter values and other configuration information
    eta, min_leaf_data, max_leaves, sst_flag, polar_flag = configurations_dict[config_id]
    eta = float(eta)
    min_leaf_data = int(min_leaf_data)
    max_leaves = int(max_leaves)
    random_state = config['81-10-9_random_states'][rep]
    
    # --------- set fixed hyperparameters --------- #
    early_stopping_rounds = 100
    max_boosting_iter = 10000
    max_depth = 15

    # file naming 
    date = datetime.today().strftime('%d-%m-%Y')
    filename = f"full_experiment_{index}_early_stopping_{early_stopping_rounds}"
    output_dir = "./"

    # ---------- load data --------- # 
    path_to_data = r'./data/complete_filtered_nao_drifter_dataset.h5'
    data = pd.read_hdf(path_to_data)

    ## separate into explanatory and response variables
    ## -------- data_config_options ----------- ##
    not_polar_with_sst_explanatory_var_labels = ['u_av','v_av','lat','lon','day_of_year','Wx','Wy','Tx','Ty','sst_x_derivative','sst_y_derivative']
    not_polar_without_sst_explanatory_var_labels = ['u_av','v_av','lat','lon','day_of_year','Wx','Wy','Tx','Ty']
    polar_with_sst_explanatory_var_labels = ['R_vel_av','arg_vel_av','lat','lon','day_of_year','R_wind_speed','arg_wind_speed','R_wind_stress','arg_wind_stress','sst_x_derivative','sst_y_derivative']
    polar_with_without_sst_explanatory_var_labels = ['R_vel_av','arg_vel_av','lat','lon','day_of_year','R_wind_speed','arg_wind_speed','R_wind_stress','arg_wind_stress']
    ##----------------------------------------- ##

    input_data_configs = [not_polar_with_sst_explanatory_var_labels,
                          polar_with_sst_explanatory_var_labels,
                          not_polar_without_sst_explanatory_var_labels,
                          polar_with_without_sst_explanatory_var_labels]

    reduced_id_options = [0,2] # non-polar
    if polar_flag:
        reduced_id_options = [1,3] #polar
    
    if sst_flag:
        explanatory_var_labels = input_data_configs[reduced_id_options[0]]
    else:
        explanatory_var_labels = input_data_configs[reduced_id_options[1]]

    response_var_labels = ['u','v']

    # -------- create base learners -------- # 
    base = DecisionTreeRegressor(
        criterion='friedman_mse',
        splitter='best',
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=min_leaf_data,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=random_state,
        max_leaf_nodes=max_leaves,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0)
    # ---------- run and save model ---------- #
    
    multivariate_ngboost = MVN_ngboost(n_estimators=max_boosting_iter,
                                       early_stopping_rounds=early_stopping_rounds,
                                       base=base,
                                       learning_rate=eta,
                                       random_state = random_state)
    multivariate_ngboost.run_model_and_save(data,explanatory_var_labels,response_var_labels,filename)
    multivariate_ngboost.save_model(filename)
    end = time.time()
    runtime = end-start

    # ------------ print useful information -------------- #
    print(f'\n Experiment Date - {date}')

    print('\n Runtime:')
    print(f'In seconds: {runtime:.2f}')
    print(f'In minutes: {runtime/60}')
    if runtime > 60**2:
        print(f'In hours: {runtime/60**2}')

    print(f'Number of Iterations: {multivariate_ngboost.best_val_loss_itr}')
    
    print(f'\n Model Hyperparameters:')
    param_names = ['learning rate', 'minimum samples per leaf', 'maximum leaves per node', 'replication number', 'Uses SST gradient', 'Uses polar (r,theta) velocity and wind data']
    params = [eta,min_leaf_data,max_leaves,rep,sst_flag,polar_flag]
    for ii,param_name in enumerate(param_names):
        print(f'{param_name}: {params[ii]}')
    print(f'Max. boosting iterations: {max_boosting_iter}')
    print(f'Max. tree depth: {max_depth}')

    print(f'\n Implementation information:')
    print(f'Early stopping rounds: {early_stopping_rounds}')
    print(f'Experiment ID: {index}')
    print(f'Configuration ID: {config_id}')
    print(f'Replication Number: {rep}')
    print(f'Random Seed Index: {config['81-10-9_random_seeds'][rep]}')
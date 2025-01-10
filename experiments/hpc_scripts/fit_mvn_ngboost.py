import sys
import os
from datetime import datetime
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))
import pandas as pd
from seaducks.models._mvn_ngboost import MVN_ngboost

if __name__=='__main__':

    # --------- set up --------- #
    _, eta, min_leaf_data, max_leaves = sys.argv
    min_leaf_data = int(min_leaf_data)
    max_leaves = int(max_leaves)
    date = datetime.today().strftime('%d-%m-%Y')

    filename = f"mvn_ngboost_eta-{eta}_min_leaf_data-{min_leaf_data}_max_leaves-{max_leaves}_date-{date}"
    output_dir = "./"

    # --------- set fixed hyperparameters --------- #
    early_stopping_rounds = 50
    max_boosting_iter = 1000
    max_depth = 15

    # ---------- load data --------- # 
    path_to_data = r'C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks\data\filtered_nao_drifters_with_sst_gradient.h5'
    data = pd.read_hdf(path_to_data)
    # add day of the year as an index (to be added to the data later)
    data['day_of_year'] = data['time'].apply(lambda t : t.timetuple().tm_yday)

    ## separate into explanatory and response variables
    explanatory_var_labels = ['u_av','v_av','lat','lon','day_of_year','Wx','Wy','Tx','Ty','sst_x_derivative','sst_y_derivative']
    response_var_labels = ['u','v']

    # ---------- run and save model ---------- #
    multivariate_ngboost = MVN_ngboost(n_estimators=max_boosting_iter,early_stopping_rounds=early_stopping_rounds)
    multivariate_ngboost.run_model_and_save(data,explanatory_var_labels,response_var_labels,filename)
    multivariate_ngboost.save_model(filename)
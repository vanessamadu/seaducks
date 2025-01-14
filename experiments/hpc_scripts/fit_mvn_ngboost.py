import sys
import os
from datetime import datetime
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))
import pandas as pd
from seaducks.models._mvn_ngboost import MVN_ngboost
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import MultivariateNormal

if __name__=='__main__':

    # --------- set up --------- #
    _, eta, min_leaf_data, max_leaves = sys.argv
    eta = float(eta)
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
    path_to_data = r'data\complete_filtered_nao_drifter_dataset.h5'
    data = pd.read_hdf(path_to_data).head(500)

    ## separate into explanatory and response variables
    explanatory_var_labels = ['u_av','v_av','lat','lon','day_of_year','Wx','Wy','Tx','Ty','sst_x_derivative','sst_y_derivative']
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
        random_state=None,
        max_leaf_nodes=max_leaves,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0)
    # ---------- run and save model ---------- #
    multivariate_ngboost = MVN_ngboost(dist = MultivariateNormal(2), n_estimators=max_boosting_iter,
                                       early_stopping_rounds=early_stopping_rounds,
                                       base=base,
                                       learning_rate=eta)
    multivariate_ngboost.run_model_and_save(data,explanatory_var_labels,response_var_labels,filename)
    multivariate_ngboost.save_model(filename)
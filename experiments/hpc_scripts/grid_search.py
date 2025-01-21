import pandas as pd
import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))

def rmse(vec1,vec2):
    return np.sqrt(np.mean(np.square(vec1-vec2)))

# initialisation
num_experiments = 2
experiment_results = pd.DataFrame(index= pd.RangeIndex(1, num_experiments + 1),columns=['RMSE'])
root_dir = 'C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks\experiments\hpc_runs\16-01-2025'

for ii in range(1,num_experiments+1):
    with open(fr'{root_dir}\model_test_data/experiment_{ii}test_data.p', 'rb') as pickle_file:
        test_data = pickle.load(pickle_file)

    with open(fr'{root_dir}/fit_models/experiment_{ii}.p', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    predicted_distribution = test_data[1]
    testing_data = test_data[0]
    locs, covs = predicted_distribution
    testing_data.loc[:,'mvn_ngb_prediction_u'] = locs[:,0]
    testing_data.loc[:,'mvn_ngb_prediction_v'] = locs[:,1]

    # multiply by 100 to convert to cm/s
    rmse_val = 100*rmse(np.array(testing_data[['u','v']]),np.array(testing_data[['mvn_ngb_prediction_u','mvn_ngb_prediction_v']]))
    experiment_results.loc[ii,'RMSE'] = rmse_val

# group by configuration index and take the mean over each group
grouped = experiment_results.groupby(np.floor(experiment_results.index/10)).mean() 
grouped.index = grouped.index.astype(int)

file_name = 'grid_search'
filehandler = open(f"{file_name}.p","wb")
pickle.dump(grouped,filehandler)
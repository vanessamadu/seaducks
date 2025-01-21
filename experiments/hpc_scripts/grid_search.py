import pandas as pd
import pickle
import sys
import os
import numpy as np
import time
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))

start = time.time()
def rmse(vec1,vec2):
    return np.sqrt(np.mean(np.square(vec1-vec2)))

# initialisation
num_experiments = 20
experiment_results = pd.DataFrame(columns=['Experiment ID','RMSE'])
root_dir = r'./'
root_dir = r'C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks\experiments\hpc_runs\16-01-2025'
date = datetime.today().strftime('%d-%m-%Y')
experiment_results['experiment ID'] = np.arange(1,num_experiments+1,dtype=int)
experiment_results['config ID'] = experiment_results['experiment ID'].apply(lambda x: int(np.floor(x/10)))

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
grouped = experiment_results.groupby('config ID').mean() 
grouped.sort_values('RMSE', ascending=True,inplace=True)
grouped.index = grouped.index.astype(int)

grouped = grouped[['RMSE']]

file_name = 'full_experiment_grid_search'
filehandler = open(f"{file_name}.p","wb")
pickle.dump(grouped,filehandler)
end = time.time()

runtime = end-start
# ------------ print useful information -------------- #
print(f'\n Experiment Date - {date}')

print('\n Runtime:')
print(f'In seconds: {runtime:.2f}')
print(f'In minutes: {runtime/60}')
if runtime > 60**2:
    print(f'In hours: {runtime/60**2}')

print(f'\n Top 5 models:')
print(grouped.head(5))


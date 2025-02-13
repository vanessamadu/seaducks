import pandas as pd
import pickle
import sys
import os
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/models'), '..')))

start = time.time()
def rmse(vec1,vec2):
    return np.sqrt(np.mean(np.square(vec1-vec2)))

# initialisation
num_experiments = 1440 
num_reps = 10
early_stopping_rounds = int(sys.argv[1])
date = str(sys.argv[2])

experiment_results = pd.DataFrame(columns=['Experiment ID','RMSE'])
root_dir = fr'/rds/general/user/vm2218/home/phd-project1/SeaDucks/seaducks/experiments/mvn_ngboost_fit_experiments/early_stopping_{early_stopping_rounds}/{date}/'
file_prefix = "experiment_"
file_suffix = f"_date_03-02-2025_early_stopping_{early_stopping_rounds}"
experiment_results['experiment ID'] = np.arange(1,num_experiments+1,dtype=int)
experiment_results['config ID'] = experiment_results['experiment ID'].apply(lambda index: int(np.floor((index-1)/num_reps)))

for ii in range(1,num_experiments+1):
    try:
        with open(fr'{root_dir}model_test_data/{file_prefix}{ii}{file_suffix}_test_data.p', 'rb') as pickle_file:
            test_data = pickle.load(pickle_file)

        with open(fr'{root_dir}fit_models/{file_prefix}{ii}{file_suffix}.p', 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        predicted_distribution = test_data[1]
        testing_data = test_data[0]
        locs, covs = predicted_distribution
        testing_data.loc[:,'mvn_ngb_prediction_u'] = locs[:,0]
        testing_data.loc[:,'mvn_ngb_prediction_v'] = locs[:,1]

        # multiply by 100 to convert to cm/s
        rmse_val = 100*rmse(np.array(testing_data[['u','v']]),np.array(testing_data[['mvn_ngb_prediction_u','mvn_ngb_prediction_v']]))
        experiment_results.loc[ii,'RMSE'] = rmse_val
    except OSError:
        print(f'Experiment {ii} not found')
        continue

# group by configuration index and take the mean over each group
grouped = experiment_results.groupby('config ID').mean() 
grouped.sort_values('RMSE', ascending=True,inplace=True)
grouped.index = grouped.index.astype(int)

grouped = grouped[['RMSE']]

file_name = f'experiment_date_{date}_early_stopping_{early_stopping_rounds}_grid_search'
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


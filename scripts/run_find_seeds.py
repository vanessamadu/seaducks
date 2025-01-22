import os
import sys
import pandas as pd
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/model_selection'), '..')))
from seaducks.model_selection import nominal_cluster_sampling

if __name__ == "__main__":
    # load data
    path_to_data = r'C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks\data\complete_filtered_nao_drifter_dataset.h5'
    data = pd.read_hdf(path_to_data)
    seeds = nominal_cluster_sampling(data, num_seeds=100)
    print(seeds)

    file_name = '81_10_9_seed_indices'
    filehandler = open(f"{file_name}.p","wb")
    pickle.dump(seeds,filehandler)
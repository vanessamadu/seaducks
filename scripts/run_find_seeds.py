import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/model_selection'), '..')))
from seaducks.model_selection import nominal_cluster_sampling

if __name__ == "__main__":
    # load data
    path_to_data = r'C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks\data\filtered_nao_drifters_with_sst_gradient.h5'
    data = pd.read_hdf(path_to_data)

    print(nominal_cluster_sampling(data, num_seeds=10))
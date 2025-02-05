import pickle
import numpy as np

def generate_samples(metric, config_1_ID, config_2_ID,
                     num_reps, root_dir,file_name_prefix,replication_ids,file_name_suffix,invalid_vals = []):
    X_1 = []
    X_2 = []
    
    for ii in range(num_reps-len(invalid_vals)):
        #load data for variable 1
        with open(fr'{root_dir}/{file_name_prefix}{replication_ids[config_1_ID][ii]}{file_name_suffix}_test_data.p', 'rb') as pickle_file:
            test_data_1 = pickle.load(pickle_file)
        # load data for variable 2
        with open(fr'{root_dir}/{file_name_prefix}{replication_ids[config_2_ID][ii]}{file_name_suffix}_test_data.p', 'rb') as pickle_file:
            test_data_2 = pickle.load(pickle_file)

        # get prediction distribution and test data
        predicted_distribution_1, predicted_distribution_2 = test_data_1[1], test_data_2[1]
        testing_data_1, testing_data_2 = test_data_1[0], test_data_2[0]
        
        locs_1, _ = predicted_distribution_1
        locs_2, _ = predicted_distribution_2

        # add realistion to sample lists
        X_1.append(metric(np.array(testing_data_1[['u','v']]),np.array(locs_1)))
        X_2.append(metric(np.array(testing_data_2[['u','v']]),np.array(locs_2)))

    return np.array(X_1),np.array(X_2)
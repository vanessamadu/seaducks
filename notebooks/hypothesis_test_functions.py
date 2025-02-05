import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt

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

def two_sample_one_sided_t_test(X_1,X_2,num_reps,X_1_name,X_2_name):
    
    # set up for t-statistic
    mean_X_1, mean_X_2 = np.mean(X_1), np.mean(X_2)
    N = num_reps

    # scipy
    t_stat, p = scipy.stats.ttest_ind(X_1, X_2, alternative = 'greater')

    # parameters for plot t-distribution
    x = np.linspace(-4, 4, 1000) # range for t-distribution
    t_dist = scipy.stats.t.pdf(x, df = 2*(N-1)) # pdf for t-distribution

    # plot t-distribution
    plt.plot(x, t_dist, label = f"t-distribution (df = {2*(N-1)})")
    plt.axvline(t_stat, color = "red", linestyle = "--", label = f"T statistic = {t_stat:.2f}")
    plt.axvline(-t_stat, color = "red", linestyle = "--")
    plt.fill_between(x, t_dist, where = (x >= abs(t_stat)), color = "red", alpha = 0.3, label = "Rejection Region (one side)")
    plt.fill_between(x, t_dist, where = (x <= -abs(t_stat)), color = "red", alpha = 0.3)
    plt.title("t-distribution and observed t statistic")
    plt.xlabel("t value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # interpret result (significance level 0.05)
    print(f"\nFor mu_X_1: mu_{X_1_name}, mu_X_2: mu_{X_2_name} \n----------------------")
    alpha = 0.05
    if p < alpha:
        print("Reject H0: mu_X_1 >= mu_X_2")
        print(f"p = {p:.2f}")
        print(f"\n mu_X_1 = {mean_X_1:.2f}")
        print(f"mean_0 = {mean_X_2:.2f}")
    else:
        if p<=1-alpha:
            print("Fail to Reject H0: mu_X_1 may be less than mu_X_2")
        else:
            print("Fail to Reject H0: mu_X_1 < mu_X_2")
        print(f"p = {p:.2f}")
        print(f"\nmu_X_1 = {mean_X_1:.2f}")
        print(f"mu_X_2 = {mean_X_2:.2f}")

def dagostino_pearson_test(X_1,X_2,X_1_name,X_2_name,alpha=0.05):
    _, p_1 = scipy.stats.normaltest(X_1)
    _, p_2 = scipy.stats.normaltest(X_2)

    test_pairs = [['X_1',p_1],['X_2',p_2]]
    print(f"\nFor mu_X_1: mu_{X_1_name}, mu_X_2: mu_{X_2_name} \n----------------------")

    for pair in test_pairs:
        if pair[1] < alpha:
            print(f"\n{pair[0]} does not follow a normal distribution.")
            print(f"p = {pair[1]:.2f}")
        else:
            print(f"\n{pair[0]} follows a normal distribution.")
            print(f"p = {pair[1]:.2f}")
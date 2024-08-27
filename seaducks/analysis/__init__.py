'''
I would like to include the following summary statistics in this file.
(For drifter sst gradients)

- mean
- standard deviation
- q-q plot
- plot histogram

and I would like to have these summary statistics for:
- the gulf stream region
- the labrador sea
- an open ocean region near the equator
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def q_q_plot_normal(data: np.ndarray):
    data_mean = np.mean(data)
    data_std = np.std(data)
    # normal distribution
    normal_dist = stats.norm(loc = data_mean, scale = data_std)
    normal_data = normal_dist.rvs(size=len(data))
    # quantiles 
    quantile_probabilities = np.linspace(0,1,50)
    normal_quantiles = np.quantile(normal_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    plt.plot(data_quantiles,normal_quantiles)
    plt.show()

def q_q_plot_laplace(data: np.ndarray):
    data_mean = np.mean(data)
    data_std = np.std(data)
    # laplace distribution
    laplace_dist = stats.laplace(loc = data_mean, scale = data_std)
    laplace_data = laplace_dist.rvs(size=len(data))
    # quantiles 
    quantile_probabilities = np.linspace(0,1,50)
    normal_quantiles = np.quantile(laplace_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    plt.plot(data_quantiles,normal_quantiles)
    plt.show()

def q_q_plot_log_laplace(data: np.ndarray):
    data_mean = np.mean(data)
    data_std = np.std(data)
    # log_laplace distribution
    log_laplace_dist = stats.loglaplace(c=1/data_std,loc = data_mean, scale = data_std)
    log_laplace_data = log_laplace_dist.rvs(size=len(data))
    # quantiles 
    quantile_probabilities = np.linspace(0,1,50)
    normal_quantiles = np.quantile(log_laplace_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    plt.plot(data_quantiles,normal_quantiles)
    plt.show()


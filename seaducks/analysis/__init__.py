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
    x = np.linspace(normal_dist.ppf(.001),normal_dist.ppf(.999))
    y=normal_dist.pdf(x,y)

    # quantiles 
    quantile_probabilities = np.linspace(0,1,50)
    normal_quantiles = np.quantile(y,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    plt.plot(data_quantiles,normal_quantiles)
    plt.show()


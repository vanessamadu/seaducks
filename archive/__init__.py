
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from seaducks.config import config

def q_q_normal(data: np.ndarray, num_quantiles: int, seed: int = config['random_state']):
    data_mean = np.mean(data)
    data_std = np.std(data)
    # normal distribution
    normal_dist = stats.norm(loc = data_mean, scale = data_std)
    normal_data = normal_dist.rvs(size=len(data),random_state=seed)
    # quantiles 
    quantile_probabilities = np.linspace(0,1,num_quantiles)
    normal_quantiles = np.quantile(normal_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    return data_quantiles,normal_quantiles
  

def q_q_laplace(data: np.ndarray, num_quantiles: int, seed: int = config['random_state']):
    data_mean = np.mean(data)
    data_std = np.std(data)
    # laplace distribution
    laplace_dist = stats.laplace(loc = data_mean, scale = data_std)
    laplace_data = laplace_dist.rvs(size=len(data),random_state=seed)
    # quantiles 
    quantile_probabilities = np.linspace(0,1,num_quantiles)
    laplace_quantiles = np.quantile(laplace_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    return data_quantiles,laplace_quantiles

def q_q_plot_log_laplace(data: np.ndarray, seed: int = config['random_state']):
    fig, ax = plt.subplots()
    ax.grid(True,linestyle='--',alpha=0.7)
    data_mean = np.mean(data)
    data_std = np.std(data)
    # log_laplace distribution
    log_laplace_dist = stats.loglaplace(c=1/data_std,loc = data_mean, scale = data_std)
    log_laplace_data = log_laplace_dist.rvs(size=len(data),random_state=seed)
    # quantiles 
    quantile_probabilities = np.linspace(0,1,50)
    normal_quantiles = np.quantile(log_laplace_data,quantile_probabilities)
    data_quantiles = np.quantile(data,quantile_probabilities)

    plt.plot(data_quantiles,normal_quantiles)
    plt.show()

def summary_plots(region_sst_grad_x:np.ndarray,region_sst_grad_y:np.ndarray, bins: int, seed: int = config['random_state'], num_quantiles: int = 100):
    
    # plot histogram and Q-Q plot
    ## configuration
    fig_region, ax_region = plt.subplots(ncols=2, figsize = (15,7.5))
    ax_hist, ax_qq = ax_region
    ## --------------- Zonal --------------- ##
    ### histogram
    ax_hist.grid(True,linestyle='--',alpha=0.7)
    ax_hist.set_xlabel("SST Gradient (K/km)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(-0.15,0.15)
    ### q-q plot
    ax_qq.grid(True,linestyle='--',alpha=0.7)
    ax_qq.set_xlabel("SST Gradient Quantiles")
    ax_qq.set_ylabel("Laplace Distribution Quantiles")
    ax_qq.set_xlim(-0.6,0.5)
    ax_qq.set_ylim(-0.6,0.5)
    ### plotting
    ax_hist.hist(region_sst_grad_x,bins = bins,density=True,color='mediumblue',histtype=u'step',ls='--', lw=1.5)
    qq = q_q_laplace(region_sst_grad_x,num_quantiles=num_quantiles,seed=seed)
    ax_qq.plot(qq[0],qq[1],color='navy',linestyle='--')
    ## -------------- Meridional -------------- ##
    ### histogram
    ax_hist.grid(True,linestyle='--',alpha=0.7)
    ax_hist.set_xlim(-0.15,0.15)
    ### q-q plot
    ax_qq.grid(True,linestyle='--',alpha=0.7)
    ax_qq.set_xlim(-0.65,0.5)
    ax_qq.set_ylim(-0.6,0.5)
    ### plotting
    ax_hist.hist(region_sst_grad_y,bins = bins,density=True,histtype=u'step',color='indianred', lw=1.5)
    qq = q_q_laplace(region_sst_grad_y,num_quantiles=num_quantiles,seed=seed)
    ax_qq.plot(qq[0],qq[1],color='firebrick')


    ax_hist.legend(['Zonal', 'Meridional'],prop={'size': 15})
    ax_qq.legend([f'Zonal (loc: {round(np.mean(region_sst_grad_x),2)}, scale: {round(np.std(region_sst_grad_x),2)})', 
                f'Meridional (loc: {round(np.mean(region_sst_grad_y),2)}, scale: {round(np.std(region_sst_grad_y),2)})'],
                prop={'size': 15})
    plt.show()

## visualisation modules
import sys
import os
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('models'), '..')))
from seaducks import assign_each_position_a_bin
from shapely.geometry import Polygon
import pandas as pd

# configure plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.style.use('ggplot')


## error metrics
def rmse(vec1,vec2):
    return 100*np.sqrt(np.mean(np.square(vec1-vec2))) # m/s -> cm/s

def mae(vec1,vec2):
    return 100*np.mean(np.abs(vec1-vec2)) # m/s -> cm/s

def maao(vec1,vec2):
    elem_wise_dot_product = np.einsum('ij,ij->i',vec1,vec2)
    normalisation = np.linalg.norm(vec1,axis=1)*np.linalg.norm(vec2,axis=1)
    return np.arccos(np.clip(
        elem_wise_dot_product/normalisation,
        -1,1
    ))
def mape(true,pred):
    return 100*np.mean(np.abs((
        true-pred
    )/true))

def rmsle(vec1,vec2):
    return np.sqrt(np.mean(np.square(
        np.log(1+vec1)-np.log(1+vec2)
        )))

# spatial plotting functions

def cuts2poly(tuple_tuple) -> Polygon:
    """

    Args:
        tuple_tuple: a tuple of pd cuts, designed for .groupby([lon_cut, lat_cut]) operations
    Returns:
         a Shapely Polygon containing the square resulting from the cuts.
    """
    lon, lat = tuple_tuple
    lon1, lon2 = lon.left, lon.right
    lat1, lat2 = lat.left, lat.right
    return Polygon(np.array([(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2)]))

def add_gridlines(ax, xlocs = [-85, -70, -55, -40],
                      ylocs = list(range(0,66,15))):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray',
                      alpha=0.5, linestyle='--', 
                      
                     )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines=False
    gl.xlocator = mticker.FixedLocator(xlocs)
    gl.ylocator = mticker.FixedLocator(ylocs)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

def plot_column(daf, column, ax, cmap='seismic', vmin_manual=None, vmax_manual=None):
    vmin, vmax =daf[column].min(), daf[column].max()
    changed_vmin, changed_vmax = False, False
    if vmin_manual is not None:
        changed_vmin = vmin!=vmin_manual
        vmin = vmin_manual
    if vmax_manual is not None:
        changed_vmax = vmax!=vmax_manual
        vmax = vmax_manual
        #norm = mpl.colors.LogNorm(vmin=daf[column].min(), vmax=daf[column].max())
    extend="neither"
    if changed_vmin:
        if changed_vmax:
            extend="both"
        else:
            extend="min"
    else:
        if changed_vmax:
            extend="max"
        
    
    if cmap == 'seismic':
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
    f = ax.get_figure()
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical',
                                   extend=extend)
    f.add_axes(cax)
    crs = ccrs.PlateCarree()
    for color, rows in daf.groupby(column):
        color_mapped = sm.cmap(norm(color))
        ax.add_geometries(
            rows["geometry"], crs=crs, facecolor=color_mapped, edgecolor=color_mapped
        )
    return cax

def add_letters(axs):
    for n, ax in enumerate(axs.flatten()):
        ax.text(-0.26, 0.5, '', transform=ax.transAxes, 
            size=13, weight='bold')

def err_df(drifter_dataset,lat_grid,lon_grid,bin_size):
    '''
    Creates a dataframe with the same indices as drifter_dataset 
    '''

    # bin the domain 
    df = assign_each_position_a_bin(drifter_dataset,lat_grid, lon_grid, bin_size=bin_size)
    vars = df.columns

    # initialise 
    ## set indices
    idx = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, observed=False)[vars].apply(lambda x:x).index
    idx = np.array([ii for ii in idx])
    idx0 = [ii[0] for ii in idx]
    idx1 = [ii[1] for ii in idx]

    plot_frame = pd.DataFrame(index=[idx0,idx1])
    plot_frame.sort_index()
    
    # with sst
    metrics_names = ['rmse','mae','maao','mape','rmsle']
    metrics = [rmse,mae,maao,mape,rmsle]

    for ii, metrics_name in enumerate(metrics_names[:-1]):
        plot_frame.loc[:,f'{metrics_name}_sst'] = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, 
                                                             observed=False)[vars].apply(lambda row: metrics[ii](row[['u','v']],np.array(row[['mvn_ngb_prediction_u_sst','mvn_ngb_prediction_v_sst']])))
        plot_frame.loc[:,f'{metrics_name}_no_sst'] = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, 
                                                           observed=False)[vars].apply(lambda row: metrics[ii](row[['u','v']],np.array(row[['mvn_ngb_prediction_u_no_sst','mvn_ngb_prediction_v_no_sst']])))
        print(f'{metrics_name} complete')

    plot_frame.loc[:,f'rmsle_sst'] = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, 
                                                             observed=False)[vars].apply(lambda row: rmsle(np.linalg.norm(row[['u','v']]),np.linalg.norm(np.array(row[['mvn_ngb_prediction_u_sst','mvn_ngb_prediction_v_sst']]))))
    plot_frame.loc[:,f'rmsle_no_sst'] = df.groupby([f"lon_bin_size_{bin_size}", f"lat_bin_size_{bin_size}"], sort=False, 
                                                           observed=False)[vars].apply(lambda row: rmsle(np.linalg.norm(row[['u','v']]),np.linalg.norm(np.array(row[['mvn_ngb_prediction_u_no_sst','mvn_ngb_prediction_v_no_sst']]))))

    plot_frame.dropna(inplace=True)
    gpd = list(map(cuts2poly,plot_frame.index))
    plot_frame['geometry']=gpd

    # add average longitudes and latitudes
    lons = np.array([(idx[0].left +idx[0].right)/2 for idx in plot_frame.index])
    lats = np.array([(idx[1].left +idx[1].right)/2 for idx in plot_frame.index])
    plot_frame['longitude'] = lons
    plot_frame['latitude'] = lats

    return plot_frame
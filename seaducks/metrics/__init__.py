''' Score functions and performance metrics '''
# Author: Vanessa Madu
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/metrics'), '..')))

from seaducks.metrics._fit_scores import R2_Score, Chi2_Statistic, Prediction_Region
from seaducks.metrics._regression import RMSE, RMSLE, MAE, MAAO

__all__ = ['chi2_stat',
           'mae',
           'maao',
           'prediction_region',
           'r2_score',
           'rmse',
           'rmsle']

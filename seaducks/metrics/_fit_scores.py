'''Goodness of fit/forecast evaluation metrics'''
# Author: Vanessa Madu

import sklearn.metrics as skm
import scipy.stats as stats
import numpy as np
from seaducks.metrics._metrics_cl import Metric

# typing
from pyvista import ArrayLike, MatrixLike
from typing import Literal
from numpy import ndarray

class R2_score(Metric):

    def __init__(self, y_true: ArrayLike | MatrixLike, y_pred: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average', 'variance_weighted'] = "uniform_average",
                 force_finite: bool =True):
        super().__init__(y_true, y_pred)

        self.multioutput = multioutput
        self.string_name = 'r2_score'
        self.sample_weight = sample_weight
        self.force_finite = force_finite

    def r2_score(self) -> (float | ndarray):
        return skm.r2_score(self.y_true, self.y_pred, 
                            sample_weight=self.sample_weight, multioutput=self.multioutput, force_finite=self.force_finite)
    
class Chi2_statistic(Metric):
    
    def __init__(self, y_true: ArrayLike | MatrixLike, y_pred: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "raw_values", axis: int=0):
        super().__init__(y_true, y_pred)

        self.multioutput = multioutput
        self.string_name = 'chi2_stat'
        self.sample_weight = sample_weight
        self.ddof = len(y_true)-1
        self.axis = axis

    def chi2_stat(self, return_p_value = False) -> (float | ndarray):
        
        chi2, p_value = stats.chisquare(self.y_true, 
        f_exp = self.y_pred, ddof = self.ddof, axis = self.axis)

        if return_p_value:
            if self.multioutput == 'raw_values':
                return chi2, p_value
            elif self.multioutput == 'uniform_average':
                return np.average(chi2, weights = self.sample_weight), 
                np.average(p_value, weights = self.sample_weight)
        else: 
            if self.multioutput == 'raw_values':
                return chi2
            elif self.multioutput == 'uniform_average':
                return np.average(chi2, weights = self.sample_weight)


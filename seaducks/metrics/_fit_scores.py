'''Goodness of fit/forecast evaluation metrics'''
# Author: Vanessa Madu

import sklearn.metrics as skm
import scipy.stats as stats
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
    pass
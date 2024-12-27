'''Scoring rules'''
# Author: Vanessa Madu
from ngboost.distns.multivariate_normal import MVNLogScore
from _metrics_cl import MVNScore
from scipy import stats

from typing import Literal
from pyvista import ArrayLike, MatrixLike
from numpy import ndarray
import numpy as np

class MVN_NLL(MVNLogScore, MVNScore):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, pred_params: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):

        super.__init__(self,y_true,pred_params)

        self.sample_weight = sample_weight
        self.multioutput = multioutput

    def mvn_log_likelihood(self):
        residuals = np.expand_dims(self.loc - self.y_true, 2)
        eta = np.squeeze(np.matmul(self.L.transpose(0, 2, 1), residuals), axis=2)

        p1 = (-0.5 * np.sum(np.square(eta), axis=1)).squeeze()
        p2 = np.sum(np.log(np.diagonal(self.L, axis1=1, axis2=2)), axis=1)

        return p1 + p2 -self.mvn_dim / 2 * np.log(2 * np.pi)

    def mvn_nll(self) -> (float | ndarray):
        if self.multioutput == 'raw_values':
            return -self.mvn_log_likelihood()
        elif self.multioutput == 'uniform_average':
            return np.average(-self.mvn_log_likelihood, weights=self.sample_weight)
 
class Prediction_Region(MVNScore):
    def __init__(self, y_true: ArrayLike | MatrixLike, pred_params: ArrayLike | MatrixLike,*,
                 alpha: ArrayLike | float = 0.90):
        super.__init__(self,y_true,pred_params)

        self.df = np.shape(y_true)[0]-1
        self.alpha = alpha
        self.critical_value = stats.chi2.ppf(self.alpha,self.df)

    @property
    def prediction_region_mask(self):
        residuals = np.expand_dims(self.loc - self.y_true, 2)
        eta = np.squeeze(np.matmul(self.L.transpose(0, 2, 1), residuals), axis=2)

        return np.matmul(eta,np.transpose(eta)) <= self.critical_value
    
    @property
    def area(self):
        pass





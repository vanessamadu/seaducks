'''Scoring rules'''
# Author: Vanessa Madu
from ngboost.distns.multivariate_normal import MVNLogScore
from ngboost.distns.multivariate_normal import get_chol_factor as get_L_matrix

from typing import Literal
from pyvista import ArrayLike, MatrixLike
from numpy import ndarray
import numpy as np

class MVN_NLL(MVNLogScore):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, pred_params: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):

        self.y_true = y_true
        self.mvn_dim = y_true.shape[0]

        self.loc = pred_params[:self.mvn_dim,:]
        self.chol_factors = pred_params[self.mvn_dim:,:]
        self.L = get_L_matrix(self.chol_factors)

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
 
class Prediction_Region():
    def __init__():
        pass
    
    @property
    def coverage(self):
        pass
    
    @property
    def area(self):
        pass





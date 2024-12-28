''' Class for all regression and fit score metrics'''
# Author: Vanessa Madu

from ngboost.distns.multivariate_normal import get_chol_factor as get_L_matrix
# typing
from pyvista import ArrayLike, MatrixLike
from typing import Literal

class Metric():
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # keyword arguments
        self.multioutput = multioutput
        self.sample_weight = sample_weight
    # read-only attributes
    @property
    def string_name(self):
        pass

    @property
    def valid_loss(self):
        pass

    @property
    def valid_risk(self):
        pass

class MVNScore():

    def __init__(self,y_true: MatrixLike | ArrayLike, pred_params: ArrayLike | MatrixLike):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            pred_params (ArrayLike | MatrixLike): _description_
        """
        
        self.y_true = y_true
        self.pred_params = pred_params
        # read-only properties
        self._mvn_dim = y_true.shape[0]
        self._loc = pred_params[:self.mvn_dim,:]
        self._chol_factors = pred_params[self.mvn_dim:,:]
        self._L = get_L_matrix(self.chol_factors)

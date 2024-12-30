''' Class for all regression and fit score metrics'''
# Author: Vanessa Madu

from ngboost.distns.multivariate_normal import get_chol_factor as get_L_matrix
import numpy as np
# typing
from pyvista import ArrayLike, MatrixLike
from typing import Literal

class Metric():
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average",
                 component_wise : None | bool = True):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&#39;raw_values&#39;, &#39;uniform_average&#39;], optional): _description_. Defaults to "uniform_average".
            component_wise (bool, optional): _description_. Defaults to True.
        """

        self.y_true, self.y_pred = is_valid_data(y_true,y_pred)
        # keyword arguments
        self.multioutput = multioutput
        self.sample_weight = sample_weight
        self.component_wise = component_wise
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
    @property
    def mvn_dim(self):
        self._mvn_dim = self.y_true.shape[0]
        return self._mvn_dim
    @property
    def loc(self):
        self._loc = self.pred_params[:self.mvn_dim,:]
        return self._loc
    @property
    def chol_factors(self):
        self._chol_factors = self.pred_params[self.mvn_dim:,:]
        return self._chol_factors
    @property
    def L(self):
        self._L = get_L_matrix(self.chol_factors)
        return self._L

def is_valid_data(y_true, y_pred):
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError('y_true and y_pred must be the same shape')
    elif len(np.shape(y_true)) == 2:
        return np.expand_dims(y_true,axis=0), np.expand_dims(y_pred,axis=0) 
    else:
        return y_true, y_pred


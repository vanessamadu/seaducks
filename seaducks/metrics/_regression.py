''' Regression performance metrics '''
# Author: Vanessa Madu

import sklearn.metrics as skm
from seaducks.metrics._metrics_cl import Metric

# typing
from typing import Literal
from pyvista import ArrayLike, MatrixLike
from numpy import ndarray
import numpy as np

class MAE(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):
        
        super().__init__(y_true,y_pred)

        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.string_name = 'mae'
        self.valid_loss = True
        self.valid_risk = True

    def mae(self) -> (float | ndarray):
        return skm.mean_absolute_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)

class MAAO(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *, 
                 sample_weight: ArrayLike | None, normalised: bool = False):
        
        super().__init__(y_true,y_pred)

        self.sample_weight = sample_weight
        self.string_name = 'maao'
        self.normalised = normalised
        # NEEDS VERIFYING
        self.valid_loss = None
        self.valid_risk = None

    def maao(self) -> (float | ndarray):
        return np.mean(self.angle_offset())

    @staticmethod
    def unit_vector(vec: ArrayLike | MatrixLike) -> ndarray:

        if len(np.shape(vec)) == 1:
            vec = vec.reshape([1,-1])

        norms = np.linalg.norm(vec, axis=1).reshape([-1,1])
        return np.divide(vec,norms)
    
    def angle_offset(self) -> (float | ndarray):

        unit_y_true = self.unit_vector(self.y_true)
        unit_y_pred = self.unit_vector(self.y_pred)

        return np.arccos(np.clip(np.sum(np.multiply(unit_y_pred,unit_y_true),axis=1),
                                 a_max=1,a_min=-1))


class RMSE(Metric):

    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):
        
        super().__init__(y_true,y_pred)

        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.string_name = 'rmse'
        self.valid_loss = True
        self.valid_risk = True
    
    def rmse(self) -> (float | ndarray):
        return skm.root_mean_squared_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)

class RMSLE(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):
        
        super().__init__(y_true,y_pred)

        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.string_name = 'rmsle'
        self.valid_loss = False
        self.valid_risk = True

    def rmsle(self) -> (float | ndarray):
        return skm.root_mean_squared_log_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)
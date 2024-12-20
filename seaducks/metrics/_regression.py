''' Regression performance metrics '''
# Author: Vanessa Madu

import sklearn.metrics as skm
from seaducks.metrics._metrics_cl import Metric

# typing
from typing import Literal
from pyvista import ArrayLike, MatrixLike
from numpy import ndarray

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
    pass

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
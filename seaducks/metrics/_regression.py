''' Regression performance metrics '''
# Author: Vanessa Madu

import sklearn.metrics as skm
from typing import Literal
from pyvista import ArrayLike, MatrixLike
from seaducks.metrics._metrics_cl import Metric

class RMSE(Metric):

    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "uniform_average"):
        
        super().__init__(y_true,y_pred)

        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.string_name = 'rmse'
    
    def rmse(self):
        return skm.root_mean_squared_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)

class RMSLE(Metric):
    pass

class MAE(Metric):
    pass

class MAAO(Metric):
    pass
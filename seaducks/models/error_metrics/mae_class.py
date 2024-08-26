from error_metrics import Error
import numpy as np
from numpy import linalg

class MAE(Error):
    def __init__(self,predictions,observations):
        self._predictions = linalg.norm(np.array(predictions),axis=1)
        self._observations = linalg.norm(np.array(observations),axis=1)
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod
    def mae(residuals_arr):
        if len(residuals_arr) == 0:
            return float('NaN')
        return np.mean(np.abs(residuals_arr))
    
    ## Error-inherited properties

    @property
    def error_type(self):
        return "mae"
    
    @property
    def error(self):
        return __class__.mae(self.residuals)
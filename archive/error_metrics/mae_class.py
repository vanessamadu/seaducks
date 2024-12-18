from seaducks.analysis.error_metrics import Error
import numpy as np
from numpy import linalg

class MAE(Error):
    def __init__(self,predictions:np.ndarray,observations:np.ndarray):
        self._predictions = linalg.norm(predictions,axis=1)
        self._observations = linalg.norm(observations,axis=1)
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod
    def mae(residuals_arr:np.ndarray) -> float:
        if len(residuals_arr) == 0:
            return np.nan
        return np.mean(np.abs(residuals_arr))
    
    ## Error-inherited properties

    @property
    def error_type(self):
        return "mae"
    
    @property
    def error(self):
        return __class__.mae(self.residuals)
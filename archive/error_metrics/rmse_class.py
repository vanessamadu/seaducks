import numpy as np
from seaducks.analysis.error_metrics import Error

class RMSE(Error):
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions).flatten()
        self._observations = np.array(observations).flatten()
    
    @property
    def predictions(self):
        return self._predictions
    @property
    def observations(self):
        return self._observations
    
    @staticmethod
    def rmse(residuals_arr: np.ndarray) -> float:
        return np.sqrt(np.mean(np.square(residuals_arr)))  

    ## Error-inherited properties
    @property
    def error_type(self):
        return "rmse"
    @property
    def error(self):
        return __class__.rmse(self.residuals)
    

        
    
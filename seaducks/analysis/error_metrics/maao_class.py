from error_metrics import Error
import numpy as np
from numpy import linalg

class MAAO(Error):
    def __init__(self,predictions,observations):
        self._predictions = np.array(predictions)
        self._observations = np.array(observations)

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @property
    def residuals(self):
        'angle between each prediction and observation'
        residuals_arr = list(np.zeros(len(self.predictions)))
        for ii in range(len(residuals_arr)):
            if (self.predictions[ii] == np.zeros(2)).all() or (self.observations[ii]==np.zeros(2)).all():
                residuals_arr[ii] = None
            else:
                obs = self.observations[ii]
                pred = self.predictions[ii]
                residuals_arr[ii] = np.arccos(np.dot(pred,obs)/(linalg.norm(pred)*linalg.norm(obs)))
        return np.array(residuals_arr)
    
    @property
    def defined_residual_indices(self):
        return np.nonzero(self.residuals != None)[0]

    @staticmethod
    def maao(defined_residuals: np.ndarray):
        # mean absolute angle offset over all residuals that are defined
        if defined_residuals.all() == None:
            return 'undefined'
        elif len(defined_residuals) == 0:
            return np.nan
        return np.mean(defined_residuals)

    ## Error-inherited properties
    @property
    def error_type(self):
        return "maao"
    
    @property
    def error(self):
        return __class__.maao(self.residuals[self.defined_residual_indices])
    
    @property
    def standard_error(self):
        if self.residuals[self.defined_residual_indices].all()==None:
            return 'undefined'
        elif len(self.residuals[self.defined_residual_indices]) == 0:
            return float('NaN') 
        return np.std(self.residuals[self.defined_residual_indices])
    
    
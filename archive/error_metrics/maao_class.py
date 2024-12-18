from seaducks.analysis.error_metrics import Error
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
            if (np.isclose(self.predictions[ii],np.zeros(2))).all() or (np.isclose(self.observations[ii],np.zeros(2))).all():
                residuals_arr[ii] = np.nan
            else:
                obs = self.observations[ii]
                pred = self.predictions[ii]

                if np.isclose(linalg.norm(pred)*linalg.norm(obs),0):
                    residuals_arr[ii]=np.nan
                else:
                    residuals_arr[ii] = np.arccos(np.dot(pred,obs)/(linalg.norm(pred)*linalg.norm(obs)))
        return np.array(residuals_arr)
    
    @property
    def defined_residual_indices(self):
        return np.nonzero(np.logical_not(np.isnan(self.residuals)))[0]

    @staticmethod
    def maao(defined_residuals: np.ndarray):
        # mean absolute angle offset over all residuals that are defined
        if np.isnan(defined_residuals).all():
            print('all values are nan')
            return np.nan
        elif len(defined_residuals) == 0:
            print('no defined residuals found')
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
        if np.isnan(self.residuals[self.defined_residual_indices]).all():
            return np.nan
        elif len(self.residuals[self.defined_residual_indices]) == 0:
            return np.nan
        return np.std(self.residuals[self.defined_residual_indices])
    
    
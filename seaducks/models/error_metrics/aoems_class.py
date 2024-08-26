# Angle Offset Error Metrics
from maao_class import MAAO
import numpy as np
import pandas as pd
from numpy import linalg

class AOEMs:
    def __init__(self,predictions: np.ndarray,observations:np.ndarray):
        self._predictions = predictions
        self._observations = observations

    @property
    def predictions(self):
        return self._predictions
    
    @property
    def observations(self):
        return self._observations
    
    @staticmethod
    def unit_vector(vector:np.ndarray) -> np.ndarray:
        return np.divide(vector,linalg.norm(vector, axis =1).reshape(-1,1))
    
    @staticmethod
    def unit_complex_conversion(vector:np.ndarray) -> complex:
        return __class__.unit_vector(vector).view(complex).squeeze()
    
    # angle offset error metrics

    @property
    def maao_all(self):
        return MAAO(self.predictions,self.observations)
    
    @property
    def anti_clockwise_rotations(self):
        return np.exp(1j*self.maao_all.residuals[self.maao_all.defined_residual_indices])
    
    ## split up into anti-clockwise offset, clockwise offset, no offset

    @property
    def no_offset_indices(self):
        if len(self.maao_all.residuals[self.maao_all.defined_residual_indices]) == 0:
            return np.array([],dtype=int)
        return np.nonzero(np.isclose(0,self.maao_all.residuals[self.maao_all.defined_residual_indices]))[0]
    
    @property
    def anti_clockwise_offset_indices(self):
        remaining_indices = np.setdiff1d(self.maao_all.defined_residual_indices,self.no_offset_indices)
        complex_pred = __class__.unit_complex_conversion(self.maao_all.predictions[remaining_indices])
        transformed_complex_obs = np.multiply(self.anti_clockwise_rotations[remaining_indices],
                                              __class__.unit_complex_conversion(self.maao_all.observations[remaining_indices]))
        if len(complex_pred) == 0:
            return np.array([],dtype=int)
        return np.nonzero(np.isclose(complex_pred,transformed_complex_obs))[0]
    
    @property
    def clockwise_offset_indices(self):
        return np.setdiff1d(self.maao_all.defined_residual_indices,np.concatenate((self.no_offset_indices,self.anti_clockwise_offset_indices)))
    
    @property
    def clockwise_anticlockwise_no_undefined_proportions(self):
        undefined_indices = np.setdiff1d(range(len(self.predictions)),self.maao_all.defined_residual_indices)
        return np.array(list(map(len,[self.anti_clockwise_offset_indices,
                    self.clockwise_offset_indices,
                    self.no_offset_indices,
                    undefined_indices
                    ])))/len(self.predictions)
    
    @property
    def maao_anticlockwise(self):
        return MAAO(self.predictions[self.anti_clockwise_offset_indices],
                    self.observations[self.anti_clockwise_offset_indices])
    
    @property
    def maao_clockwise(self):
        return MAAO(self.predictions[self.clockwise_offset_indices],
                    self.observations[self.clockwise_offset_indices])
    
    ## Error-inherited property overwrites
    
    @property
    def error_summary(self):
        err_metrics = [self.maao_all,self.maao_anticlockwise,self.maao_clockwise]
        err_metric_names = ["MAAO over all Angle Offsets","MAAO for Anticlockwise Offsets", "MAAO for Clockwise Offsets"]
        err_summary = {}
        for ii in range(len(err_metrics)):
            if (err_metrics[ii].error) == 'undefined':
                err = 'undefined'
            else:
                err = np.rad2deg(err_metrics[ii].error)
            if err_metrics[ii].uncertainty == 'undefined':
                uncert = 'undefined'
            else:
                uncert = np.rad2deg(err_metrics[ii].uncertainty)

            err_summary[f"{err_metric_names[ii]} Error: {err_metrics[ii].error_type}"]=f"{err} degrees"
            err_summary[f"{err_metric_names[ii]} Standard Error: {err_metrics[ii].uncertainty_type}"] = f"{uncert} degrees"
        err_summary["Proportion of Anticlockwise/Clockwise/No/Undefined Angle Offset"]=f"{self.clockwise_anticlockwise_no_undefined_proportions*100}%"
        return pd.Series(err_summary)
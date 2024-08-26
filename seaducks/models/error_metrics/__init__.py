# general error class
from abc import ABC,abstractproperty
import numpy as np
import pandas as pd

class Error(ABC):

    @abstractproperty
    def predictions(self):
        pass

    @abstractproperty
    def observations(self):
        pass
    
    @abstractproperty
    def error_type(self):
        pass

    @abstractproperty
    def error(self):
        pass
    
    @property
    def uncertainty(self):
        if len(self.residuals) ==0:
            return float('NaN')
        return np.std(self.residuals)
    
    @property
    def uncertainty_type(self):
        return "std"
    
    @property
    def error_summary(self):
        return pd.Series({self.error_type:self.error,
                self.uncertainty_type:self.uncertainty})

    @property
    def residuals(self):
        return np.array([pred-obs for pred,obs in zip(
            self.predictions,
            self.observations)])
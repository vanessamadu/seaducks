'description: hierarchical model framework for the construction of my ocean drifter velocity model'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%% SET UP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
##### import packages #####
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Model(ABC):
    '''
    this class will be the parent class of all ocean models that we will be using
    and it will define attributes and methods common to all specific model classes.
    '''
    def __init__(self,data: pd.DataFrame):
        self._data = data

    @staticmethod
    def check_coordinates(lon:float, lat:float):
        limits = {"lat":90.,"lon":180.}
        values = {"lat":lat,"lon":lon}

        for coord in values.keys():
            try:
                float(values[coord])
            except:
                raise ValueError(f"{coord} must be a real number")
            finally:
                if np.abs(values[coord])>limits[coord]:
                    raise ValueError(f"{coord} must be between -{limits[coord]} and {limits[coord]}")
                
    @property
    def data(self):
        return self._data
    
    @property
    def observations(self):
        return np.array(self.data[["u","v"]])
    
    @abstractmethod
    def predictions(self):
        pass
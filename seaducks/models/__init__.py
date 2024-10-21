##### import packages #####
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Model(ABC):
    '''
    A parent class for the ocean surface current models featured in SeaDucks.
    
    Attributes
    ----------
    data: pd.DataFrame
        A dataset containing drifter data and satellite (derived) data interpolated to drifter
        positions.
    '''
    def __init__(self,data: pd.DataFrame):
        ''''
        Initialises Model with data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to assign to the instance
        '''
        self._data = data

    @staticmethod
    def check_coordinates(lon:float, lat:float):
        '''
        Checks that coordinates are valid.

        Parameters
        ----------
        lon: float
            longitude 
        lat: float
            latitude

        Raises
        ------
        ValueError: when
            - lat or lon are not floats
            - |lat|>90 and |lon|>180
        '''
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
        '''
        [immutable property] Gets value of data. 

        Returns
        -------
        pd.DataFrame:
            The data associated with the instance.
        '''
        return self._data
    
    @property
    def observations(self):
        '''
        [immutable property] Gets drifter velocities.

        Returns
        -------
        np.array:
        The values of (u,v) for each drifter.
        '''
        return np.array(self.data[["u","v"]])
    
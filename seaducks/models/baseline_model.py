from models import Model
# import packages
import numpy as np

class BathtubModel(Model):
    '''
    A class for a bathtub model predicting all velocities to be zero at all positions and 
    for all times. 

    Attributes
    ----------
    data: pd.DataFrame
        (inherited) A dataset containing drifter data and satellite (derived) data interpolated to drifter
        positions.
    '''
    
    @staticmethod
    def bathtub(lon:float,lat:float) -> np.ndarray:
        '''
        Returns zero for all latitudes and longitudes.

        Parameters
        ----------
        lon: float
            longitude 
        lat: float
            latitude

        Returns
        -------
        np.ndarray
            A two-array of zeros.
        '''
        __class__.check_coordinates(lon,lat)
        return np.zeros(2)
    
    # Model-inherited properties
    @property
    def predictions(self):
        return np.array([__class__.bathtub(lon,lat) for lon,lat in np.array(self.data[["lon","lat"]])])



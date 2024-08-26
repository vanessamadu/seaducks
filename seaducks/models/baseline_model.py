from models import Model
# import packages
import numpy as np

class BathtubModel(Model):
    '''benchmark model - predicts all velocities to be zero at all positions and for all times.'''
    
    @staticmethod
    def bathtub(lon,lat):
        __class__.check_coordinates(lon,lat)
        return np.zeros(2)
    
    # Model-inherited properties
    @property
    def predictions(self):
        return np.array([__class__.bathtub(lon,lat) for lon,lat in np.array(self.data[["lon","lat"]])])



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

class SBRModel(Model):
    '''benchmark model: predicts velocities according to a steady solid body rotation model.'''

    def __init__(self,data):
        super().__init__(data)
        self.f0 = 7.27e-4 # coriolis parameter at 30 degrees N

    @staticmethod
    def sbr(lon:float,lat:float,f0:float):
        __class__.check_coordinates(lon,lat)
        return np.array([-f0*(lat+42),f0*(lon-30)])

    @property
    def f0(self):
        return self._f0
    
    @f0.setter
    def f0(self,val):
        try:
            float(val)
        except:
            raise ValueError("coriolis parameter, f0, must be a real number")
        self._f0 = val

    @property
    def predictions(self):
        return np.array([__class__.sbr(lon,lat,self.f0) for lon,lat in np.array(self.data[["lon","lat"]])])
    
class FixedCurrentModel(Model):
    '''benchmark model: predicts all drifter velocities to be the average velocity across the 
       drifter data'''
    def __init__(self, data):
        super().__init__(data)
        self.av_drifter_velocity = None
  
    @staticmethod
    def fixedcurrent(lon:float,lat:float,current):
        __class__.check_coordinates(lon,lat)
        return current

    @property
    def av_drifter_velocity(self):
        return self._av_drifter_velocity
    
    @av_drifter_velocity.setter
    def av_drifter_velocity(self,val):
        if val is None:
            self._av_drifter_velocity = np.mean(np.array(self.data[["u","v"]]),axis=0)
        else:
            self._av_drifter_velocity = val


    @property
    def predictions(self):
        return np.array([__class__.fixedcurrent(lon,lat,self.av_drifter_velocity)
                         for lon,lat in np.array(self.data[["lon","lat"]])])


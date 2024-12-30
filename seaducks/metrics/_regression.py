""" Regression performance metrics """
# Author: Vanessa Madu
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/metrics'), '..')))
from seaducks.metrics._metrics_cl import Metric
import sklearn.metrics as skm
# typing
from typing import Literal
from pyvista import ArrayLike, MatrixLike
from numpy import ndarray
import numpy as np

class MAE(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal["raw_values", "uniform_average"] = "uniform_average",
                 component_wise : bool = True):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal["raw_values", "uniform_average"], optional): _description_. Defaults to "uniform_average".
        """
        super().__init__(y_true,y_pred,sample_weight=sample_weight,multioutput=multioutput,component_wise=component_wise)
    # read-only attributes
    @property
    def string_name(self):
        self._string_name = "mae"
        return self._string_name
    @property
    def valid_loss(self):
        self._valid_loss = True
        return self._valid_loss
    @property
    def valid_risk(self):
        self._valid_risk = True
        return self._valid_risk

    def mae(self) -> (float | ndarray):

        if self.component_wise:
            axis = 1
        else:
            axis = (1,2)
        raw_mae_values = np.average(np.abs(self.y_true-self.y_pred),weights = self.sample_weight,axis=axis)

        if self.multioutput == 'raw_values':
            return raw_mae_values
        elif self.multioutput == 'uniform_average':
            return np.mean(raw_mae_values, axis=0)

class MAAO(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *, 
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal["raw_values", "uniform_average"] = "uniform_average", 
                 normalize: bool = False):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
            sample_weight (ArrayLike | None): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&quot;raw_values&quot;, &quot;uniform_average&quot;], optional): _description_. Defaults to "uniform_average".
            normalize (bool, optional): _description_. Defaults to False.
        """
        super().__init__(y_true,y_pred,sample_weight=sample_weight,multioutput=multioutput, component_wise=None)
        
        self.normalize = normalize
    #read-only attributes
    @property
    def string_name(self):
        self._string_name = "maao"
        return self._string_name
    @property
    def valid_loss(self):
        # NEEDS VERIFYING
        self._valid_loss = None
        return self._valid_loss
    @property
    def valid_risk(self):
        # NEEDS VERIFYING
        self._valid_risk = None
        return self._valid_risk

    def maao(self) -> (float | ndarray):
        if self.multioutput == "raw_values":
            return np.average(self.angle_offset(),
            weights=self.sample_weight,axis=0)
        elif self.multioutput == "uniform_average":
            return np.average(self.angle_offset(),
            weights=self.sample_weight)

    @staticmethod
    def unit_vector(vec: ArrayLike | MatrixLike) -> ndarray:

        if len(np.shape(vec)) == 1:
            vec = vec.reshape([1,-1])

        norms = np.linalg.norm(vec, axis=1).reshape([-1,1])
        return np.divide(vec,norms)
    
    def angle_offset(self) -> (float | ndarray):

        unit_y_true = self.unit_vector(self.y_true)
        unit_y_pred = self.unit_vector(self.y_pred)

        if self.normalize:
            return np.arccos(np.clip(np.sum(np.multiply(unit_y_pred,unit_y_true),axis=1),
                                 a_max=1,a_min=-1))/np.pi
        else:
            if self.normalize:
                return np.arccos(np.clip(np.sum(np.multiply(unit_y_pred,unit_y_true),axis=1),
                                 a_max=1,a_min=-1))

class RMSE(Metric):

    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal["raw_values", "uniform_average"] = "uniform_average",
                 component_wise : bool = True):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&quot;raw_values&quot;, &quot;uniform_average&quot;], optional): _description_. Defaults to "uniform_average".
        """
        super().__init__(y_true,y_pred, sample_weight=sample_weight, multioutput=multioutput, component_wise=component_wise)
        # read-only attributes
    @property
    def string_name(self):
        self._string_name = "rmse"
        return self._string_name
    @property
    def valid_loss(self):
        self._valid_loss = True
        return self._valid_loss
    @property
    def valid_risk(self):
        self._valid_risk = True
        return self._valid_risk
    
    def rmse(self) -> (float | ndarray):
        return skm.root_mean_squared_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)

class RMSLE(Metric):
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal["raw_values", "uniform_average"] = "uniform_average",
                 component_wise : bool = True):
        """_summary_

        Args:
            y_true (MatrixLike | ArrayLike): _description_
            y_pred (MatrixLike | ArrayLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&quot;raw_values&quot;, &quot;uniform_average&quot;], optional): _description_. Defaults to "uniform_average".
        """
        super().__init__(y_true,y_pred, sample_weight=sample_weight, multioutput=multioutput, component_wise=component_wise)
        # read-only attributes
    @property
    def string_name(self):
        self._string_name = "rmsle"
        return self._string_name
    @property
    def valid_loss(self):
        self._valid_loss = False
        return self._valid_loss
    @property
    def valid_risk(self):
        self._valid_risk = True
        return self._valid_risk

    def rmsle(self) -> (float | ndarray):
        return skm.root_mean_squared_log_error(self.y_true, self.y_pred, 
                                           sample_weight=self.sample_weight, multioutput=self.multioutput)
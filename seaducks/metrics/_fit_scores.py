'''Goodness of fit/forecast evaluation metrics'''
# Author: Vanessa Madu

import sklearn.metrics as skm
import scipy.stats as stats
import numpy as np
from seaducks.metrics._metrics_cl import MVNScore
from scipy import stats
from scipy.special import gamma

# typing
from pyvista import ArrayLike, MatrixLike
from typing import Literal
from numpy import ndarray

class R2_score():

    def __init__(self, y_true: ArrayLike | MatrixLike, y_pred: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average', 'variance_weighted'] = "uniform_average",
                 force_finite: bool =True):
        """_summary_

        Args:
            y_true (ArrayLike | MatrixLike): _description_
            y_pred (ArrayLike | MatrixLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&#39;raw_values&#39;, &#39;uniform_average&#39;, &#39;variance_weighted&#39;], optional): _description_. Defaults to "uniform_average".
            force_finite (bool, optional): _description_. Defaults to True.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # keyword arguments
        self.multioutput = multioutput
        self.sample_weight = sample_weight
        self.force_finite = force_finite
    # read-only properties
    @property
    def string_name(self):
        self._string_name = 'r2_score'
        return self._string_name
        
    def r2_score(self) -> (float | ndarray):
        return skm.r2_score(self.y_true, self.y_pred, 
                            sample_weight=self.sample_weight, multioutput=self.multioutput, force_finite=self.force_finite)
    
class Chi2_statistic():
    
    def __init__(self, y_true: ArrayLike | MatrixLike, y_pred: ArrayLike | MatrixLike, *,
                 sample_weight: ArrayLike | None = None, multioutput: ArrayLike | Literal['raw_values', 'uniform_average'] = "raw_values", axis: int=0):
        """_summary_

        Args:
            y_true (ArrayLike | MatrixLike): _description_
            y_pred (ArrayLike | MatrixLike): _description_
            sample_weight (ArrayLike | None, optional): _description_. Defaults to None.
            multioutput (ArrayLike | Literal[&#39;raw_values&#39;, &#39;uniform_average&#39;], optional): _description_. Defaults to "raw_values".
            axis (int, optional): _description_. Defaults to 0.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # keyword arguments
        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.axis = axis
    # read-only attributes
    @property
    def string_name(self):
        self._string_name = 'chi2_stat'
        return self._string_name
    @property
    def ddof(self):
        self._ddof = np.shape(self.y_true)[0]-1
        return self._ddof
        
    def chi2_stat(self, return_p_value = False) -> (float | ndarray):
        
        chi2, p_value = stats.chisquare(self.y_true, 
        f_exp = self.y_pred, ddof = self.ddof, axis = self.axis)

        if return_p_value:
            if self.multioutput == 'raw_values':
                return chi2, p_value
            elif self.multioutput == 'uniform_average':
                return np.average(chi2, weights = self.sample_weight),
            np.average(p_value, weights = self.sample_weight)
        else: 
            if self.multioutput == 'raw_values':
                return chi2
            elif self.multioutput == 'uniform_average':
                return np.average(chi2, weights = self.sample_weight)

class Prediction_Region(MVNScore):
    def __init__(self, y_true: ArrayLike | MatrixLike, pred_params: ArrayLike | MatrixLike,*,
                 alpha: ArrayLike | float = 0.90):
        super.__init__(self,y_true,pred_params)

        self.alpha = alpha
        self.string_name = 'prediction_region'
        
        self._critical_value = None
        self._area = None
        self._coverage = None
        self._df = None

    # hidden attirbutes
    @property
    def critical_value(self):
        self._critical_value = stats.chi2.ppf(self.alpha,self.df)
        return self._critical_value
    
    @property
    def df(self):
        self._df = np.shape(self.y_true)[0]-1
        return self._df
    
    @property
    def coverage(self):
        pr = self.prediction_region_mask()
        self._coverage = np.sum(pr,axis=(0,1))/np.prod(np.shape(pr)[0:2])
        return self._coverage

    @property
    def area(self):
        num_data_points = np.shape(self.y_true)[0]
        multiplier = ((2*np.pi)**(num_data_points/2))/(num_data_points*gamma(num_data_points/2))
        if len(np.shape(self.L)) == 3:
            eigenvalues = np.array([np.diag(l) for l in self.L])
            self._area = multiplier*self.critical_value*(np.divide(1,np.sum(eigenvalues,axis=1)))
        else:
            eigenvalues = np.array(np.diag(self.L))
            self._area = multiplier*self.critical_value*(1/np.sum(eigenvalues))
        return self._area
        

    def prediction_region_mask(self):
        residuals = np.expand_dims(self.loc - self.y_true, 2)
        eta = np.squeeze(np.matmul(self.L.transpose(0, 2, 1), residuals), axis=2)

        return np.matmul(eta,np.transpose(eta)) <= self.critical_value


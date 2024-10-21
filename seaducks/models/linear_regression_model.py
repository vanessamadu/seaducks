from seaducks.models import Model
#import packages
import numpy as np
import pandas as pd
from numpy import linalg

class LinearRegressionModel(Model):
    '''
    A class implementing a linear regression model.

    Attributes
    ----------
    data: pd.DataFrame
        (inherited) A dataset containing drifter data and satellite (derived) data interpolated to drifter
        positions.
    covariate_labels: list
        List of covariates used in the linear regression model.
    param_estimate: np.ndarray
        The parameter, beta, of the linear regression model.
    '''
    
    def __init__(self,data: pd.DataFrame,covariate_labels:list):
        '''
        Initialises the LinearRegressionModel with data and a list of covariate labels.

        Parameters
        ----------

        data: pd.DataFrame
            The data to assign to the instance.
        covariate_labels: list
            The covariate labels to assign to the instance.
        '''
        super().__init__(data)
        self.covariate_labels = covariate_labels
        self.param_estimate = None

    @staticmethod
    def lr(X:np.ndarray,beta:np.ndarray) -> np.ndarray:
        '''
        Returns a linear regression model prediction for covariate values, X.

        Parameters
        ----------
        X: np.ndarray
            An array of covariate values.
        beta: np.ndarray
            The LinearRegressionModel parameter

        Returns
        -------
        np.ndarray
            A linear regression prediction.

        Raises
        ------
        ValueError: when
        - Covariates, X, and the linear regression parameter, beta, have incompatible dimensions.
        '''
        try:
            pred = np.matmul(X,beta)
            return pred
        except:
            ValueError(f"incompatible dimensions covariates : {X.shape}; parameters: {beta.shape}")

    @property
    def design(self):
        '''
        Gets the design matrix associated with the training data
        
        Returns
        -------
        np.ndarray
            The design matrix.

        Raises
        ------
        KeyError: when
        - Labels in the covariate labels list are not found in the data.
        '''
        try:
            design_matrix = np.array(self.data.loc[:,self.covariate_labels])
            return design_matrix
        except:
            raise KeyError("Covariate(s) were not found in the dataset")
    
    def calculate_param_estimate(self):
        '''
        Returns the least squares model parameter estimate.

        Returns
        -------
        np.ndarray:
            The linear regression parameter estimate.
        '''
        lstsq_estimate = linalg.lstsq(self.design,
                                       np.array(self.data.loc[:,["u","v"]]),
                                       rcond=None)
        self.param_estimate= lstsq_estimate[0]

    
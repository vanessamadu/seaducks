from models import Model
#import packages
import numpy as np
from numpy import linalg

class LinearRegressionModel(Model):
    
    def __init__(self,data,covariate_labels):
        super().__init__(data)
        self.covariate_labels = covariate_labels
        self.param_estimate = None

    @staticmethod
    def lr(X,beta):
        '''returns a prediction for the linear regression model'''
        try:
            pred = np.matmul(X,beta)
            return pred
        except:
            ValueError(f"incompatible dimensions covariates : {X.shape}; parameters: {beta.shape}")

    @property
    def design(self):
        '''returns the design matrix associated with the training data'''
        try:
            design_matrix = np.array(self.data.loc[:,self.covariate_labels])
            return design_matrix
        except:
            raise KeyError("Covariate(s) were not found in the dataset")
    
    def calculate_param_estimate(self):
        '''returns least squares parameter estimate'''
        lstsq_estimate = linalg.lstsq(self.design,
                                       np.array(self.data.loc[:,["u","v"]]),
                                       rcond=None)
        self.param_estimate= lstsq_estimate[0]

    @property
    def predictions(self):
        ' return prediction for each vector of covariates for seen data'
        if self.param_estimate is None:
            self.calculate_param_estimate()
        pred = __class__.lr(self.design,
                                   self.param_estimate)
        return pred

    
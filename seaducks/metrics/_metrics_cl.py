''' Class for all regression and fit score metrics'''
# Author: Vanessa Madu

from pyvista import ArrayLike, MatrixLike
from ngboost.distns.multivariate_normal import get_chol_factor as get_L_matrix

class Metric():
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike):
        self.y_true = y_true
        self.y_pred = y_pred

        self.multioutput = None
        self.string_name = None
        self.sample_weight = None
        self.valid_loss = None
        self.valid_risk = None

class MVNScore():

    def __init__(self,y_true: MatrixLike | ArrayLike, pred_params: ArrayLike | MatrixLike):
        
        self.y_true = y_true
        self.pred_params = pred_params

        self.mvn_dim = y_true.shape[0]
        
        self.loc = pred_params[:self.mvn_dim,:]
        self.chol_factors = pred_params[self.mvn_dim:,:]
        self.L = get_L_matrix(self.chol_factors)

''' Class for all regression and fit score metrics'''
# Author: Vanessa Madu

from pyvista import ArrayLike, MatrixLike

class Metric():
    
    def __init__(self, y_true: MatrixLike | ArrayLike, y_pred: MatrixLike | ArrayLike):
        self.y_true = y_true
        self.y_pred = y_pred

        self.multioutput = None
        self.string_name = None
        self.sample_weight = None

    def error(self):
        pass
    
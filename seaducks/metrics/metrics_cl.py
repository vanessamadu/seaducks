''' Class for all regression and fit score metrics'''
# Author: Vanessa Madu

import numpy as np

class Metric():
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

        self.multioutput = None
        self.string_name = None
        self.sample_weight = None

    def error(self):
        pass
    
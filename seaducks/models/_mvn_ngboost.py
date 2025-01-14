import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/model_selection'), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('seaducks/config'), '..')))

from ngboost import NGBRegressor
from ngboost.scores import LogScore
from ngboost.learners import default_tree_learner

import ngboost.distns
import ngboost.scores
import ngboost.learners

import pickle
from seaducks.config import config
from seaducks.model_selection import train_test_validation_split_ids

class MVN_ngboost(NGBRegressor):
    def __init__(self, dist: ngboost.distns,*,
                 score: ngboost.scores = LogScore, 
                 base: ngboost.learners = default_tree_learner, natural_gradient: bool = True,
                 n_estimators: int = 500, learning_rate: float = 0.01, minibatch_frac: float = 1.0,
                 col_sample: float =1.0, verbose: bool =True, verbose_eval: int = 100, tol: float = 1e-4,
                 random_state: None | int = config['81-10-9_random_states'][0], validation_fraction: float = 0.09, early_stopping_rounds: int = None):
        
        super().__init__(
        Dist=dist,
        Score=score,
        Base=base,
        natural_gradient=natural_gradient,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=minibatch_frac,
        col_sample=col_sample,
        verbose=verbose,
        verbose_eval=verbose_eval,
        tol=tol,
        random_state=random_state,
        validation_fraction=validation_fraction,
        early_stopping_rounds=early_stopping_rounds)

    def save_model(self, file_name):
        filehandler = open(f"{file_name}.p","wb")
        pickle.dump(self,filehandler)

    def run_model_and_save(self,data,explanatory_var_labels,response_var_labels,file_name,
                                test_frac = 0.10, validation_frac = 0.09,
                                shuffle = True, stratify = None):
        # Note to self: write for clustered and non-clustered sampling
        ids = (data.groupby('id').size()).index
        train_ids, test_ids, val_ids = train_test_validation_split_ids(ids,
                                test_frac = test_frac, validation_frac = validation_frac, 
                                random_state = self.random_state, shuffle = shuffle, stratify = stratify)
        
        training_data = data.query('id in @train_ids')
        testing_data = data.query('id in @test_ids')
        validation_data = data.query('id in @val_ids')

        training_explanatory_vars = training_data[explanatory_var_labels]
        testing_explanatory_vars = testing_data[explanatory_var_labels]
        validation_explanatory_vars = validation_data[explanatory_var_labels]

        training_response_vars = training_data[response_var_labels]
        validation_response_vars = validation_data[response_var_labels]

        output_file = file_name +'test_data'

        if not os.path.isfile(output_file):
            self.fit(training_explanatory_vars, training_response_vars, 
                     validation_explanatory_vars, validation_response_vars, 
                     early_stopping_rounds=self.early_stopping_rounds)
            
        preds = [testing_data, self.scipy_distribution(testing_explanatory_vars,cmat_output = True)]

        filehandler = open(f"{output_file}.p","wb")
        pickle.dump(preds,filehandler)

    def scipy_distribution(self, explanatory_vars, cmat_output=False):
        preds = self.pred_dist(explanatory_vars, max_iter=self.best_val_loss_itr)
        if cmat_output:
            out = [preds.mean(), preds.cov]
        else:
            out = preds.scipy_distribution()
        return out


        
        

    
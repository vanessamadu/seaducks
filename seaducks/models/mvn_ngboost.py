from ngboost.ngboost import NGBoost
from ngboost.scores import LogScore
from ngboost.distns import MultivariateNormal
from ngboost.learners import default_tree_learner

import ngboost.distns
import ngboost.scores
import ngboost.learners

class MVN_ngboost(NGBoost):
    def __init__(self, *,
                 dist: ngboost.distns = MultivariateNormal, score: ngboost.scores = LogScore, 
                 base: ngboost.learners = default_tree_learner, natural_gradient: bool = True,
                 n_estimators: int = 500, learning_rate: float = 0.01, minibatch_frac: float = 1.0,
                 col_sample: float =1.0, verbose: bool =True, verbose_eval: int = 100, tol: float = 1e-4,
                 random_state: None | int = 42, validation_fraction: float = 0.09, early_stopping_rounds: int = None):
        
        super.__init__(
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
        
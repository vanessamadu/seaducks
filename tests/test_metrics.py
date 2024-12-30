import pytest
# random seed
from numpy.random import MT19937, RandomState, SeedSequence

from numpy import ndarray
import numpy as np
# metrics
from seaducks.metrics import MAE
# random values
from random_values import y_true_10_10_2,y_pred_10_10_2

rs = RandomState(MT19937(SeedSequence(0)))
# -------------- Helpers -------------- #

def synthetic_data(size: float | tuple) -> ndarray:
    y_pred = np.random.uniform(size = size)
    y_true = np.random.uniform(size = size,low=1.0,high=5.0)
    return y_true, y_pred

# -------------- Regression Metrics -------------- #
@pytest.mark.regression

## === MAE === #

@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("size",[
    (0,0),
    (0,10),
    (10,2),
    (100,2),
    (10,10,2),
    (100,100,2)
])
def test_mae_attributes_are_correctly_assigned(size):
    y_true, y_pred = synthetic_data(size)
    trial_mae = MAE(y_true, y_pred)
    attributes = [trial_mae.string_name, trial_mae.valid_loss, 
            trial_mae.valid_risk, trial_mae.sample_weight, trial_mae.multioutput, trial_mae.component_wise]
    assert attributes == ['mae',True,True,None,'uniform_average',True]

@pytest.mark.regression
@pytest.mark.mean_absolute_error
def test_mae_expand_dims():
    n = 100
    y_true, y_pred = synthetic_data((n,2))
    trial_mae = MAE(y_true,y_pred)
    assert (np.shape(trial_mae.y_pred) == np.shape(trial_mae.y_true)) and np.shape(trial_mae.y_pred) == (1,n,2)

@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("size",[
    [(10,2),(50,2)],
    [(10,2),(10,3)],
    [(100,2),(50,100,2)],
    [(10,10,2), (10,100,2)],
    [(10,10,2), (10,10,3)],
    [(100,10,2), (10,100,2)]
])
def test_mae_wrong_dimensions_raises_exception(size):
    size1, size2 = size
    y_true, _ = synthetic_data(size1)
    _, y_pred = synthetic_data(size2)
    with pytest.raises(ValueError) as info:
        trial_mae = MAE(y_true,y_pred)

@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("flags",[
    ['raw_values', True],
    ['raw_values', False],
    ['uniform_average', True],
    ['uniform_average', False]
])
def test_mae_output_is_correct_shape(flags):
    multioutput, component_wise = flags
    m = 10
    n = 100
    size = (m,n,2)
    y_true, y_pred = synthetic_data(size)
    trial_mae = MAE(y_true,y_pred,
                    multioutput=multioutput,component_wise=component_wise)
    
    if multioutput == 'raw_values':
        if component_wise:
            asserted_shape = (m,2)
        else:
            asserted_shape = (m,)
    elif multioutput == 'uniform_average':
        if component_wise:
            asserted_shape = (2,)
        else:
            asserted_shape = ()
    assert np.shape(trial_mae.mae()) == asserted_shape

@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("flags",[
    ['raw_values', True],
    ['raw_values', False],
    ['uniform_average', True],
    ['uniform_average', False]
])
def test_mae_correct_values(flags):
    multioutput,component_wise = flags
    trial_mae = MAE(y_true_10_10_2,y_pred_10_10_2,
                    multioutput=multioutput,component_wise=component_wise)
    is_array = True
    if multioutput == 'raw_values':
        if component_wise:
            val = np.array([[2.3925327 , 2.28000627],
       [2.67197687, 2.21757877],
       [3.04211805, 2.86112999],
       [3.11798521, 3.27704392],
       [2.45549111, 1.68638292],
       [2.25407313, 2.48026473],
       [2.49960347, 2.18696163],
       [2.17345121, 2.62379441],
       [2.45668352, 2.17777013],
       [2.74935874, 3.06567944]])
        else:
            val = np.array([2.33626948, 2.44477782, 2.95162402, 3.19751456, 2.07093701,
       2.36716893, 2.34328255, 2.39862281, 2.31722682, 2.90751909])
    elif multioutput == 'uniform_average':
        if component_wise:
            val = np.array([2.5813274 , 2.48566122])
        else:
            val = 2.533494309837941
            is_array = False
    
    if is_array:
        assert (np.isclose(val,trial_mae.mae())).all()
    else:
        assert np.isclose(val,np.array(trial_mae.mae()))

@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("flags",[
    ['raw_values', True],
    ['raw_values', False],
    ['uniform_average', True],
    ['uniform_average', False]
])
def test_correctly_handles_zeros(flags):
    multioutput,component_wise = flags
    n = 10
    m = 100
    size = (m,n,2)
    y_true,y_pred = [np.zeros(size),np.zeros(size)]
    trial_mae = MAE(y_true,y_pred,
                    multioutput=multioutput,component_wise=component_wise)
    is_array = True
    
    if multioutput == 'uniform_average':
        if not component_wise:
            is_array = False
    
    if is_array:
        assert (np.isclose(0.0,trial_mae.mae())).all()
    else:
        assert np.isclose(0.0,np.array(trial_mae.mae()))

    
@pytest.mark.regression
@pytest.mark.mean_absolute_error
@pytest.mark.parametrize("flags",[
    ['raw_values', True],
    ['raw_values', False],
    ['uniform_average', True],
    ['uniform_average', False]
])
def test_mae_negative_values(flags):
    multioutput,component_wise = flags
    trial_mae = MAE(-y_true_10_10_2,y_pred_10_10_2,
                    multioutput=multioutput,component_wise=component_wise)
    is_array = True
    if multioutput == 'raw_values':
        if component_wise:
            val = np.array([[3.13109083, 3.37225728],
       [3.6278231 , 3.08104839],
       [3.82498433, 4.03814793],
       [3.9836221 , 4.25563999],
       [3.51975134, 2.57149536],
       [3.34239646, 3.43896852],
       [3.40779386, 3.2594528 ],
       [3.24428394, 3.39015019],
       [3.43981467, 2.92227029],
       [3.91259216, 4.26655196]])
        else:
            val = np.array([3.25167406, 3.35443575, 3.93156613, 4.11963105, 3.04562335,
       3.39068249, 3.33362333, 3.31721707, 3.18104248, 4.08957206])
    elif multioutput == 'uniform_average':
        if component_wise:
            val = np.array([3.54341528, 3.45959827])
        else:
            val = 3.5015067763711736
            is_array = False
    
    if is_array:
        assert (np.isclose(val,trial_mae.mae())).all()
    else:
        assert np.isclose(val,np.array(trial_mae.mae()))

@pytest.mark.regression
@pytest.mark.parametrize("maao",[

])
def test_maao(maao):
    pass

@pytest.mark.regression
@pytest.mark.parametrize("rmse",[

])
def test_rmse(rmse):
    pass

@pytest.mark.regression
@pytest.mark.parametrize("rmsle",[

])
def test_rmsle(rmsle):
    pass


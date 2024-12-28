import pytest

"""
Parameterize example
@pytest.mark.parametrize("true",[
    1+1 == 2,
    True,
    type(5) is int
])
def test_test(true):
    assert true
"""
# -------------- Regression Metrics -------------- #

@pytest.mark.regression
@pytest.mark.parametrize("mae",[

])
def test_mae(mae):
    pass

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


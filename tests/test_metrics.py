import pytest

@pytest.mark.regression
@pytest.mark.parametrize("true",[
    1+1 == 2,
    True,
    type(5) is int
])
def test_test(true):
    assert true
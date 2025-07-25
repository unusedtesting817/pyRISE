import pytest
import numpy as np
from pyrise2.dsge_var import dsge_var

def test_dsge_var():
    dsge_solution = None
    var_data = np.random.rand(100, 3)
    n_lags = 4
    dsge_var_solution = dsge_var(dsge_solution, var_data, n_lags)
    assert dsge_var_solution["dsge_var_solution"] == "dummy"

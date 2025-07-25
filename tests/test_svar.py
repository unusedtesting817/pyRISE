import pytest
import numpy as np
from pyrise2.svar import svar

def test_svar():
    var_residuals = np.random.rand(100, 3)
    restrictions = None
    svar_solution = svar(var_residuals, restrictions)
    assert svar_solution["svar_solution"] == "dummy"

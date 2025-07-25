import pytest
import numpy as np
from pyrise2.forecast import forecast, conditional_forecast

def test_forecast():
    solution = None
    initial_conditions = np.array([1.0, 2.0, 3.0])
    n_periods = 10
    f = forecast(solution, initial_conditions, n_periods)
    assert f.shape == (n_periods, len(initial_conditions))

def test_conditional_forecast():
    solution = None
    initial_conditions = np.array([1.0, 2.0, 3.0])
    n_periods = 10
    controlled_vars = {"x": np.zeros(10)}
    f = conditional_forecast(solution, initial_conditions, n_periods, controlled_vars)
    assert f.shape == (n_periods, len(initial_conditions))

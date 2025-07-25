import pytest
import numpy as np
from pyrise2.estimation import mle

class LinearRegressionModel:
    def log_likelihood(self, beta, data):
        X, y = data
        return -0.5 * np.sum((y - X @ beta)**2)

def test_mle_linear_regression():
    # Generate some data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    beta_true = np.array([1.5, -2.0])
    y = X @ beta_true + 0.1 * np.random.randn(100)
    data = (X, y)

    # Estimate the parameters
    model = LinearRegressionModel()
    beta_initial = np.array([0.0, 0.0])
    beta_estimated = mle(model, data, beta_initial)

    # Check that the estimated parameters are close to the true parameters
    np.testing.assert_allclose(beta_estimated, beta_true, atol=0.1)

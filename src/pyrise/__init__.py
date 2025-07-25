__version__ = "0.1.0"

from .core.model import Model
from .solvers.perturbation import PerturbationSolver
from .utils.validation import validate_discount_factor, validate_transition_matrix
from .utils.reproducibility import configure_reproducible_environment

def load_model(model_file: str) -> "Model":
    """
    Load a DSGE model from YAML specification.
    """
    return Model.from_file(model_file)

def solve(model: "Model", **kwargs) -> "PerturbationSolver":
    """
    Solve a regime-switching DSGE model using perturbation methods.
    """
    solver = PerturbationSolver(**kwargs)
    return solver.solve(model)

def estimate(model: "Model", data, method: str = "NUTS", **kwargs):
    """
    Estimate model parameters using Bayesian methods.
    """
    from .estimation import BayesianEstimator
    estimator = BayesianEstimator(method=method, **kwargs)
    return estimator.estimate(model, data)

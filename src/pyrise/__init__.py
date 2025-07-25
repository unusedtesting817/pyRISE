__version__ = "0.1.0"

def load_model(model_file: str):
    """
    Load a DSGE model from YAML specification.
    """
    pass

def solve(model, **kwargs):
    """
    Solve a regime-switching DSGE model using perturbation methods.
    """
    pass

def estimate(model, data, method: str = "NUTS", **kwargs):
    """
    Estimate model parameters using Bayesian methods.
    """
    pass

class MarkovKernel:
    pass

class ThresholdKernel:
    pass

import pandas as pd
from pyrise.solvers.perturbation import Solution

def irf(solution: Solution, shock_name: str, shock_size: float = 1.0, horizon: int = 40, regime: int = 0) -> pd.DataFrame:
    """
    Compute impulse response functions.
    """
    responses = solution.compute_irf(shock_name, shock_size, horizon, regime)
    return pd.DataFrame(responses)

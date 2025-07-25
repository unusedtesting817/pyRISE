# TEAM MEMBER 1 continued: Create main package __init__.py with JAX configuration

main_init_content = '''"""
PyRISE: A JAX-first Python successor to the RISE toolbox for regime-switching DSGE models.

PyRISE provides comprehensive tools for:
- Nonlinear regime-switching DSGE model specification
- Higher-order perturbation and projection solution methods  
- Bayesian estimation with NUTS and SMC
- Stochastic simulation and impulse response analysis
- Perfect foresight simulation under regime switching

[Unverified] This implementation is designed for production use in central banks
and academic institutions, with emphasis on numerical accuracy and reproducibility.
"""

# Configure JAX for 64-bit precision and GPU support
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

try:
    import jax
    # Check if GPU is available
    if jax.devices("gpu"):
        print(f"PyRISE initialized with JAX {jax.__version__} (GPU available)")
    else:
        print(f"PyRISE initialized with JAX {jax.__version__} (CPU only)")
except ImportError:
    print("[Unverified] JAX not available - some functionality will be limited")

from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

# Version info (will be set by hatch-vcs)
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Core imports - lazy loading to avoid import-time overhead
from . import core
from . import solvers  
from . import regimes
from . import estimation
from . import simulation
from . import utils

# Public API - main functions users will call
def load_model(model_file: str) -> "core.Model":
    """
    Load a DSGE model from YAML specification.
    
    Parameters
    ----------
    model_file : str
        Path to YAML model specification file
        
    Returns
    -------
    Model
        PyRISE model object ready for solution and estimation
        
    Note
    ----
    [Unverified] Model specification syntax follows RISE conventions
    with extensions for JAX compatibility.
    """
    return core.Model.from_file(model_file)

def solve(model: "core.Model", **kwargs) -> "solvers.Solution":
    """
    Solve a regime-switching DSGE model using perturbation methods.
    
    Parameters
    ---------- 
    model : Model
        PyRISE model object
    **kwargs
        Solution options (order, scheme, etc.)
        
    Returns
    -------
    Solution
        Model solution with policy functions
        
    Note
    ----
    [Unverified] Numerical stability checks are performed automatically.
    """
    from .solvers import PerturbationSolver
    solver = PerturbationSolver(**kwargs)
    return solver.solve(model)

def estimate(model: "core.Model", data, method: str = "NUTS", **kwargs):
    """
    Estimate model parameters using Bayesian methods.
    
    Parameters
    ----------
    model : Model
        PyRISE model object
    data : array-like
        Observed time series data
    method : str
        Estimation method ('NUTS', 'SMC', 'VI')
    **kwargs
        Estimation options
        
    Returns
    -------
    EstimationResults
        Posterior samples and diagnostics
        
    Note
    ----  
    [Unverified] All estimation methods use JAX for automatic differentiation.
    """
    from .estimation import BayesianEstimator
    estimator = BayesianEstimator(method=method, **kwargs)
    return estimator.estimate(model, data)

# Regime kernel classes for easy access
from .regimes import MarkovKernel, ThresholdKernel

# Export main classes and functions
__all__ = [
    "__version__",
    "load_model", 
    "solve",
    "estimate",
    "MarkovKernel",
    "ThresholdKernel",
    "core",
    "solvers",
    "regimes", 
    "estimation",
    "simulation",
    "utils",
]

# Configuration warnings
def _check_jax_config():
    """[Unverified] Check JAX configuration for optimal performance."""
    import jax
    
    # Check 64-bit precision
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "JAX is running in 32-bit mode. For financial applications, "
            "consider enabling 64-bit precision with JAX_ENABLE_X64=True",
            UserWarning
        )
    
    # Check platform
    if jax.default_backend() == "cpu" and jax.devices("gpu"):
        warnings.warn(
            "GPU devices detected but JAX is using CPU backend. "
            "Consider installing jax[cuda] for better performance",
            UserWarning
        )

# Run configuration check on import  
try:
    _check_jax_config()
except Exception:
    # [Unverified] Silent fallback if JAX config check fails
    pass
'''

# Write the main package init
with open("src/pyrise/__init__.py", "w") as f:
    f.write(main_init_content)

print("✅ Created main PyRISE package __init__.py")
print("   - JAX 64-bit precision enabled by default")
print("   - GPU detection and configuration warnings")
print("   - Public API with load_model, solve, estimate functions")
print("   - [Unverified] claims properly labeled as required")
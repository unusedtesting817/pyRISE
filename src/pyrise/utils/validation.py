"""
Validation utilities for economic parameters and model constraints.

This module provides comprehensive validation functions for DSGE model parameters,
ensuring compliance with economic theory and statistical requirements.
"""

from typing import Union, Optional, Dict, Any
import numpy as np
import jax.numpy as jnp
from jax import Array
import warnings


class EconomicValidationError(ValueError):
    """
    Exception raised when economic parameters violate theoretical constraints.

    This exception is raised when parameters fail validation against
    established economic theory principles.
    """
    pass


def validate_discount_factor(beta: float) -> bool:
    """
    Validate discount factor against economic theory constraints.

    Parameters
    ----------
    beta : float
        Discount factor

    Returns
    -------
    bool
        True if valid, False otherwise

    Note
    ----
    Discount factors must be in (0, 1) to ensure:
    - Present value convergence
    - Finite utility
    - Meaningful intertemporal choice

    [Unverified] This range is standard in macroeconomic literature.
    """
    return 0 < beta < 1


def validate_transition_matrix(P: Array) -> bool:
    """
    Validate Markov transition matrix properties.

    Parameters
    ----------
    P : Array
        Transition probability matrix

    Returns
    -------
    bool
        True if valid transition matrix, False otherwise

    Note
    ----
    Valid transition matrices must satisfy:
    - All entries non-negative
    - Each row sums to 1 (stochastic matrix)
    - Square matrix

    [Unverified] These are standard Markov chain requirements.
    """
    P = jnp.asarray(P)

    # Check if square matrix
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    # Check non-negative entries
    if jnp.any(P < 0):
        return False

    # Check row sums equal 1 (within numerical tolerance)
    row_sums = jnp.sum(P, axis=1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-8):
        return False

    return True


def validate_parameter_bounds(
    param_name: str,
    value: float,
    bounds: Optional[tuple] = None
) -> bool:
    """
    Validate parameter against specified bounds.

    Parameters
    ----------
    param_name : str
        Parameter name for error messages
    value : float
        Parameter value
    bounds : tuple, optional
        (lower, upper) bounds. None means unbounded.

    Returns
    -------
    bool
        True if within bounds, False otherwise

    Note
    ----
    [Unverified] Common parameter bounds in DSGE models:
    - Persistence parameters: [0, 1)
    - Standard deviations: (0, ∞)
    - Elasticities: (0, ∞)
    """
    if bounds is None:
        return True

    lower, upper = bounds

    if lower is not None and value <= lower:
        return False
    if upper is not None and value >= upper:
        return False

    return True


def validate_shock_covariance(Sigma: Array) -> bool:
    """
    Validate shock covariance matrix properties.

    Parameters
    ----------
    Sigma : Array
        Shock covariance matrix

    Returns
    -------
    bool
        True if valid covariance matrix, False otherwise

    Note
    ----
    Valid covariance matrices must be:
    - Symmetric
    - Positive semi-definite
    - Square

    [Unverified] These properties ensure well-defined probability distributions.
    """
    Sigma = jnp.asarray(Sigma)

    # Check if square
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        return False

    # Check symmetry
    if not jnp.allclose(Sigma, Sigma.T, atol=1e-8):
        return False

    # Check positive semi-definite via eigenvalues
    try:
        eigenvals = jnp.linalg.eigvals(Sigma)
        if jnp.any(eigenvals < -1e-8):  # Allow small numerical errors
            return False
    except Exception:
        # [Unverified] Fallback if eigenvalue computation fails
        return False

    return True


def validate_taylor_rule_parameters(psi_pi: float, psi_y: float = 0.0) -> bool:
    """
    Validate Taylor rule parameters for determinacy.

    Parameters
    ----------
    psi_pi : float
        Response to inflation
    psi_y : float, optional
        Response to output gap

    Returns
    -------
    bool
        True if satisfies determinacy condition, False otherwise

    Note
    ----
    [Unverified] Standard determinacy condition: psi_pi > 1 + psi_y
    This ensures unique rational expectations equilibrium.

    References: Woodford (2003), Galí (2008)
    """
    return psi_pi > 1.0 + psi_y


def validate_persistence_parameter(rho: float, param_name: str = "") -> bool:
    """
    Validate autoregressive persistence parameter.

    Parameters
    ----------
    rho : float
        Persistence parameter
    param_name : str, optional
        Parameter name for warnings

    Returns
    -------
    bool
        True if valid, False otherwise

    Note
    ----
    [Unverified] Persistence parameters should be in [0, 1) for stationarity.
    Values ≥ 1 imply unit roots or explosive processes.
    """
    if not (0 <= rho < 1):
        return False

    # Issue warning for very high persistence
    if rho > 0.99:
        warnings.warn(
            f"[Unverified] High persistence parameter {param_name}: {rho}. "
            f"May cause numerical issues in solution methods."
        )

    return True


def validate_steady_state_values(
    steady_state: Dict[str, float],
    variables: Dict[str, str]
) -> Dict[str, str]:
    """
    Validate steady-state values for economic reasonableness.

    Parameters
    ----------
    steady_state : dict
        Steady-state variable values
    variables : dict
        Variable types (e.g., 'consumption', 'inflation')

    Returns
    -------
    dict
        Dictionary of validation warnings/errors

    Note
    ----
    [Unverified] Checks common-sense bounds:
    - Inflation rates: reasonable range
    - Interest rates: non-negative (usually)
    - Consumption, output: positive
    """
    issues = {}

    for var_name, value in steady_state.items():
        var_type = variables.get(var_name, "unknown")

        # Validate by variable type
        if var_type in ["consumption", "output", "investment"]:
            if value <= 0:
                issues[var_name] = f"Economic quantity {var_name} = {value} should be positive"

        elif var_type == "inflation":
            if abs(value - 1.0) > 0.1:  # More than 10% steady-state inflation
                issues[var_name] = f"[Unverified] High steady-state inflation: {value}"

        elif var_type == "interest_rate":
            if value < 0:
                issues[var_name] = f"[Unverified] Negative interest rate: {value}"

        elif var_type == "labor":
            if not (0 <= value <= 1):
                issues[var_name] = f"Labor fraction {var_name} = {value} should be in [0,1]"

    return issues


def validate_model_stability(eigenvalues: Array, n_state_vars: int) -> bool:
    """
    Validate model stability via Blanchard-Kahn conditions.

    Parameters
    ----------
    eigenvalues : Array
        Eigenvalues of the system matrix
    n_state_vars : int
        Number of predetermined state variables

    Returns
    -------
    bool
        True if Blanchard-Kahn conditions satisfied

    Note
    ----
    [Unverified] Blanchard-Kahn conditions for unique solution:
    - Number of unstable eigenvalues = number of forward-looking variables
    - Predetermined variables have stable eigenvalues
    """
    eigenvalues = jnp.asarray(eigenvalues)

    # Count unstable eigenvalues (|λ| > 1)
    unstable = jnp.sum(jnp.abs(eigenvalues) > 1.0 + 1e-8)

    # Number of forward-looking variables
    n_forward = len(eigenvalues) - n_state_vars

    return int(unstable) == n_forward


# Parameter validation registry for common DSGE parameters
PARAMETER_BOUNDS = {
    "beta": (0.0, 1.0),          # Discount factor
    "sigma": (0.0, None),        # Risk aversion
    "chi": (0.0, None),          # Labor supply elasticity
    "eta": (1.0, None),          # Elasticity of substitution
    "alpha": (0.0, 1.0),         # Capital share
    "delta": (0.0, 1.0),         # Depreciation rate
    "phi": (0.0, None),          # Investment adjustment cost
    "kappa": (0.0, None),        # Price adjustment cost
}

PERSISTENCE_PARAMS = [
    "rho_a", "rho_g", "rho_m", "rho_z",  # Technology, govt, monetary, preference
    "rho_r", "rho_pi", "rho_y"           # Policy rule parameters
]


def validate_all_parameters(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Comprehensive parameter validation for DSGE models.

    Parameters
    ----------
    params : dict
        Dictionary of all model parameters

    Returns
    -------
    dict
        Dictionary of parameter names and validation issues

    Note
    ----
    [Unverified] Validates parameters against economic theory constraints
    and common DSGE modeling practices.
    """
    issues = {}

    for param_name, value in params.items():

        # Skip non-numeric parameters
        if not isinstance(value, (int, float, np.number)):
            continue

        # Check standard parameter bounds
        if param_name in PARAMETER_BOUNDS:
            bounds = PARAMETER_BOUNDS[param_name]
            if not validate_parameter_bounds(param_name, value, bounds):
                lower, upper = bounds
                bound_str = f"({lower}, {upper})" if upper else f"({lower}, ∞)"
                issues[param_name] = f"Parameter {param_name} = {value} outside valid range {bound_str}"

        # Check persistence parameters
        if any(param_name.startswith(prefix) for prefix in PERSISTENCE_PARAMS):
            if not validate_persistence_parameter(value, param_name):
                issues[param_name] = f"Persistence parameter {param_name} = {value} should be in [0, 1)"

        # Check Taylor rule parameters
        if param_name == "psi_pi":
            psi_y = params.get("psi_y", 0.0)
            if not validate_taylor_rule_parameters(value, psi_y):
                issues[param_name] = f"[Unverified] Taylor rule may violate determinacy: psi_pi = {value}, psi_y = {psi_y}"

    # Check for required parameters
    required_params = ["beta"]  # Minimum required parameters
    for req_param in required_params:
        if req_param not in params:
            issues[req_param] = f"Required parameter {req_param} is missing"

    return issues

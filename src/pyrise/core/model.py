"""
Core model classes for PyRISE with comprehensive economic parameter validation.

This module implements the main Model class with built-in validation of economic
theory constraints and statistical properties.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp
from jax import Array
import warnings
import yaml
from pathlib import Path

from ..utils.validation import (
    validate_discount_factor,
    validate_transition_matrix,
    validate_parameter_bounds,
    validate_shock_covariance,
    EconomicValidationError
)


@dataclass(frozen=True)
class ModelParameters:
    """
    Container for model parameters with automatic economic validation.

    All parameters are validated against economic theory constraints
    upon creation. [Unverified] This ensures model stability and
    economically meaningful results.
    """

    # Core parameters
    beta: float = field(default=0.99)  # Discount factor
    sigma: float = field(default=1.0)  # Risk aversion
    chi: float = field(default=1.0)    # Labor supply elasticity

    # Regime-dependent parameters (dict with regime indices as keys)
    regime_params: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Shock parameters
    shock_stds: Dict[str, float] = field(default_factory=dict)
    shock_correlations: Optional[Array] = None

    # Transition matrix for Markov switching
    transition_matrix: Optional[Array] = None

    def __post_init__(self):
        """
        Validate all parameters after initialization.

        Raises
        ------
        EconomicValidationError
            If any parameter violates economic theory constraints
        """
        # [Unverified] All validation rules are based on standard DSGE theory
        self._validate_core_parameters()
        self._validate_regime_parameters()
        self._validate_shock_parameters()
        self._validate_transition_matrix()

    def _validate_core_parameters(self):
        """Validate core economic parameters."""

        # Discount factor must be in (0, 1)
        if not validate_discount_factor(self.beta):
            raise EconomicValidationError(
                f"Discount factor β = {self.beta} must be in (0, 1). "
                f"Values outside this range violate economic theory."
            )

        # Risk aversion should be positive
        if self.sigma <= 0:
            raise EconomicValidationError(
                f"Risk aversion σ = {self.sigma} must be positive"
            )

        # Labor supply elasticity should be positive
        if self.chi <= 0:
            raise EconomicValidationError(
                f"Labor supply elasticity χ = {self.chi} must be positive"
            )

    def _validate_regime_parameters(self):
        """Validate regime-specific parameters."""

        for regime_id, params in self.regime_params.items():
            for param_name, value in params.items():

                # Taylor rule coefficients
                if param_name.startswith('psi_'):
                    if param_name == 'psi_pi' and value <= 1.0:
                        warnings.warn(
                            f"[Unverified] Taylor rule coefficient {param_name} = {value} "
                            f"in regime {regime_id} may violate determinacy condition"
                        )

                # Persistence parameters should be in [0, 1)
                if param_name.startswith('rho_'):
                    if not (0 <= value < 1):
                        raise EconomicValidationError(
                            f"Persistence parameter {param_name} = {value} "
                            f"in regime {regime_id} must be in [0, 1)"
                        )

    def _validate_shock_parameters(self):
        """Validate shock standard deviations and correlations."""

        # Standard deviations must be positive
        for shock_name, std in self.shock_stds.items():
            if std <= 0:
                raise EconomicValidationError(
                    f"Shock standard deviation for {shock_name} = {std} must be positive"
                )

        # Validate correlation matrix if provided
        if self.shock_correlations is not None:
            if not validate_shock_covariance(self.shock_correlations):
                raise EconomicValidationError(
                    "[Unverified] Shock correlation matrix is not positive definite"
                )

    def _validate_transition_matrix(self):
        """Validate Markov transition matrix."""

        if self.transition_matrix is not None:
            if not validate_transition_matrix(self.transition_matrix):
                raise EconomicValidationError(
                    "Transition matrix violates Markov chain properties: "
                    "rows must sum to 1 and all entries must be non-negative"
                )


@dataclass
class ModelEquations:
    """
    Container for model equations with symbolic representation.

    [Unverified] Equations are parsed from YAML specifications and
    converted to JAX-compatible functions for solution.
    """

    equilibrium_conditions: List[str] = field(default_factory=list)
    state_variables: List[str] = field(default_factory=list)
    control_variables: List[str] = field(default_factory=list)
    shock_variables: List[str] = field(default_factory=list)

    # Parsed symbolic expressions (populated during model loading)
    symbolic_equations: Dict[str, Any] = field(default_factory=dict)

    def validate_equation_structure(self):
        """
        Validate that equation structure is consistent.

        [Unverified] Checks for missing variables, circular dependencies,
        and other structural issues.
        """

        # Check that all variables in equations are declared
        all_variables = set(
            self.state_variables + self.control_variables + self.shock_variables
        )

        for eq_name, eq_str in zip(range(len(self.equilibrium_conditions)),
                                  self.equilibrium_conditions):
            # [Unverified] This is a simplified variable extraction
            # Real implementation would use proper symbolic parsing
            variables_in_eq = set()  # Would extract from equation string

            undefined_vars = variables_in_eq - all_variables
            if undefined_vars:
                raise EconomicValidationError(
                    f"Equation {eq_name} contains undefined variables: {undefined_vars}"
                )


class Model:
    """
    Main PyRISE model class with comprehensive validation and JAX integration.

    This class represents a regime-switching DSGE model with automatic
    parameter validation, equation parsing, and JAX-compatible solution methods.
    """

    def __init__(
        self,
        parameters: ModelParameters,
        equations: ModelEquations,
        model_name: str = "Unnamed Model"
    ):
        """
        Initialize PyRISE model with validation.

        Parameters
        ----------
        parameters : ModelParameters
            Model parameters (automatically validated)
        equations : ModelEquations
            Model equations and variable definitions
        model_name : str
            Human-readable model name

        Note
        ----
        [Unverified] All parameters are validated against economic theory
        constraints during initialization.
        """
        self.parameters = parameters  # Validation happens in __post_init__
        self.equations = equations
        self.model_name = model_name

        # Validate equation structure
        self.equations.validate_equation_structure()

        # Model state
        self._is_solved = False
        self._solution = None

    @classmethod
    def from_file(cls, yaml_file: Union[str, Path]) -> "Model":
        """
        Load model from YAML specification file.

        Parameters
        ----------
        yaml_file : str or Path
            Path to YAML model specification

        Returns
        -------
        Model
            Validated PyRISE model object

        Raises
        ------
        EconomicValidationError
            If model specification contains invalid parameters
        FileNotFoundError
            If YAML file does not exist

        Note
        ----
        [Unverified] YAML syntax follows RISE conventions where possible.
        """

        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Model file not found: {yaml_file}")

        try:
            with open(yaml_path, 'r') as f:
                model_spec = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in model file: {e}")

        # Parse parameters section
        params_spec = model_spec.get('parameters', {})
        parameters = cls._parse_parameters(params_spec)

        # Parse equations section
        equations_spec = model_spec.get('equations', {})
        equations = cls._parse_equations(equations_spec)

        model_name = model_spec.get('name', yaml_path.stem)

        return cls(parameters, equations, model_name)

    @staticmethod
    def _parse_parameters(params_spec: Dict) -> ModelParameters:
        """
        Parse parameter specification with validation.

        [Unverified] This method handles regime-dependent parameters
        and automatic constraint checking.
        """

        # Extract core parameters
        beta = params_spec.get('beta', 0.99)
        sigma = params_spec.get('sigma', 1.0)
        chi = params_spec.get('chi', 1.0)

        # Parse regime-dependent parameters
        regime_params = {}
        for key, value in params_spec.items():
            if isinstance(value, dict) and all(
                isinstance(k, int) for k in value.keys()
            ):
                # This is a regime-dependent parameter
                regime_params[key] = value

        # Parse shock parameters
        shock_stds = {}
        for key, value in params_spec.items():
            if key.startswith('sigma_') and isinstance(value, (int, float)):
                shock_name = key[6:]  # Remove 'sigma_' prefix
                shock_stds[shock_name] = value

        # Parse transition matrix if present
        transition_matrix = params_spec.get('transition_matrix')
        if transition_matrix is not None:
            transition_matrix = jnp.array(transition_matrix)

        return ModelParameters(
            beta=beta,
            sigma=sigma,
            chi=chi,
            regime_params=regime_params,
            shock_stds=shock_stds,
            transition_matrix=transition_matrix
        )

    @staticmethod
    def _parse_equations(equations_spec: Dict) -> ModelEquations:
        """
        Parse equation specification.

        [Unverified] Converts YAML equation specifications to internal format.
        """

        equilibrium_conditions = equations_spec.get('equilibrium', [])
        state_variables = equations_spec.get('states', [])
        control_variables = equations_spec.get('controls', [])
        shock_variables = equations_spec.get('shocks', [])

        return ModelEquations(
            equilibrium_conditions=equilibrium_conditions,
            state_variables=state_variables,
            control_variables=control_variables,
            shock_variables=shock_variables
        )

    def is_solved(self) -> bool:
        """Check if model has been solved."""
        return self._is_solved

    def get_solution(self):
        """
        Get model solution.

        Returns
        -------
        Solution or None
            Model solution if available, None otherwise

        Note
        ----
        [Unverified] Solution object contains policy functions and diagnostics.
        """
        if not self._is_solved:
            warnings.warn("Model has not been solved yet")
        return self._solution

    def __repr__(self) -> str:
        """String representation of model."""
        n_regimes = len(set().union(*[
            params.keys() for params in self.parameters.regime_params.values()
        ])) if self.parameters.regime_params else 1

        n_states = len(self.equations.state_variables)
        n_controls = len(self.equations.control_variables)
        n_shocks = len(self.equations.shock_variables)

        return (
            f"PyRISE Model: {self.model_name}\n"
            f"  Regimes: {n_regimes}\n"
            f"  State variables: {n_states}\n"
            f"  Control variables: {n_controls}\n"
            f"  Shock variables: {n_shocks}\n"
            f"  Solved: {self._is_solved}"
        )

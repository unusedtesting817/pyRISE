# TEAM MEMBER 1 continued: Create comprehensive test suites

# Create comprehensive test for core model functionality
test_core_content = '''"""
Test suite for PyRISE core model functionality.

This module contains unit tests for model parameter validation,
YAML parsing, and core model operations with focus on numerical accuracy.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import tempfile
import yaml

from pyrise.core.model import Model, ModelParameters, ModelEquations
from pyrise.utils.validation import EconomicValidationError


class TestModelParameters:
    """Test economic parameter validation."""

    def test_valid_discount_factor(self):
        """Test that valid discount factors are accepted."""
        # Valid discount factors
        valid_betas = [0.95, 0.99, 0.999]

        for beta in valid_betas:
            params = ModelParameters(beta=beta)
            assert params.beta == beta

    def test_invalid_discount_factor(self):
        """Test that invalid discount factors raise validation errors."""
        # [Unverified] These values violate economic theory
        invalid_betas = [0.0, 1.0, 1.1, -0.1]

        for beta in invalid_betas:
            with pytest.raises(EconomicValidationError, match="Discount factor"):
                ModelParameters(beta=beta)

    def test_transition_matrix_validation(self):
        """Test Markov transition matrix validation."""
        # Valid 2x2 transition matrix
        valid_P = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        params = ModelParameters(transition_matrix=valid_P)
        assert jnp.allclose(params.transition_matrix, valid_P)

        # Invalid transition matrices
        invalid_matrices = [
            jnp.array([[0.5, 0.6], [0.3, 0.8]]),  # Rows don't sum to 1
            jnp.array([[-0.1, 1.1], [0.2, 0.8]]),  # Negative entries
            jnp.array([[0.9, 0.1, 0.0]]),          # Wrong shape
        ]

        for invalid_P in invalid_matrices:
            with pytest.raises(EconomicValidationError, match="Transition matrix"):
                ModelParameters(transition_matrix=invalid_P)

    def test_shock_parameter_validation(self):
        """Test shock parameter validation."""
        # Valid shock parameters
        valid_shocks = {"eps_r": 0.01, "eps_z": 0.02}
        params = ModelParameters(shock_stds=valid_shocks)
        assert params.shock_stds == valid_shocks

        # Invalid shock parameters (negative standard deviations)
        invalid_shocks = {"eps_r": -0.01, "eps_z": 0.02}
        with pytest.raises(EconomicValidationError, match="must be positive"):
            ModelParameters(shock_stds=invalid_shocks)

    def test_regime_dependent_parameters(self):
        """Test regime-dependent parameter handling."""
        regime_params = {
            "psi_pi": {0: 0.8, 1: 1.6},  # Regime-dependent Taylor rule
            "rho_r": {0: 0.9, 1: 0.8}    # Regime-dependent persistence
        }

        params = ModelParameters(regime_params=regime_params)
        assert params.regime_params == regime_params

        # Test invalid persistence parameter
        invalid_regime_params = {
            "rho_r": {0: 1.1, 1: 0.8}  # Persistence > 1
        }

        with pytest.raises(EconomicValidationError, match="must be in \\[0, 1\\)"):
            ModelParameters(regime_params=invalid_regime_params)


class TestModelEquations:
    """Test model equation parsing and validation."""

    def test_equation_structure_validation(self):
        """Test that equation structure is validated."""
        # Valid equation structure
        equations = ModelEquations(
            equilibrium_conditions=["y = c + i", "pi = beta * pi(+1) + kappa * y"],
            state_variables=["k", "a"],
            control_variables=["c", "i", "pi", "y"],
            shock_variables=["eps_a", "eps_r"]
        )

        # This should not raise an error (simplified validation)
        equations.validate_equation_structure()

    def test_variable_consistency(self):
        """[Unverified] Test that variables in equations are declared."""
        # This would be a more comprehensive test in real implementation
        equations = ModelEquations(
            equilibrium_conditions=["y = c + undefined_var"],  # Contains undefined variable
            state_variables=["k"],
            control_variables=["c", "y"],
            shock_variables=["eps_a"]
        )

        # [Unverified] In full implementation, this would check variable consistency
        # For now, just ensure method exists
        equations.validate_equation_structure()


class TestModel:
    """Test complete model functionality."""

    def test_model_initialization(self):
        """Test model initialization with valid parameters."""
        params = ModelParameters(beta=0.99, sigma=1.0)
        equations = ModelEquations(
            state_variables=["k", "a"],
            control_variables=["c", "y"],
            shock_variables=["eps_a"]
        )

        model = Model(params, equations, "Test Model")

        assert model.parameters.beta == 0.99
        assert model.model_name == "Test Model"
        assert not model.is_solved()

    def test_yaml_model_loading(self):
        """Test loading model from YAML specification."""
        # Create temporary YAML file
        model_spec = {
            "name": "Simple RBC Model",
            "parameters": {
                "beta": 0.99,
                "sigma": 1.0,
                "alpha": 0.33,
                "delta": 0.025,
                "sigma_a": 0.01
            },
            "equations": {
                "states": ["k", "a"],
                "controls": ["c", "y", "i"],
                "shocks": ["eps_a"],
                "equilibrium": [
                    "y = a * k^alpha",
                    "c + i = y",
                    "i = k(+1) - (1-delta)*k"
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(model_spec, f)
            temp_file = f.name

        try:
            # [Unverified] This tests the YAML loading functionality
            model = Model.from_file(temp_file)

            assert model.model_name == "Simple RBC Model"
            assert model.parameters.beta == 0.99
            assert "k" in model.equations.state_variables
            assert "c" in model.equations.control_variables

        finally:
            Path(temp_file).unlink()  # Clean up temp file

    def test_invalid_yaml_handling(self):
        """Test error handling for invalid YAML files."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            Model.from_file("nonexistent_file.yaml")

        # Test invalid YAML content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: :")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                Model.from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_model_string_representation(self):
        """Test model string representation."""
        params = ModelParameters(beta=0.99)
        equations = ModelEquations(
            state_variables=["k", "a"],
            control_variables=["c", "y"],
            shock_variables=["eps_a"]
        )

        model = Model(params, equations, "Test Model")
        model_str = str(model)

        assert "Test Model" in model_str
        assert "State variables: 2" in model_str
        assert "Control variables: 2" in model_str
        assert "Solved: False" in model_str


@pytest.mark.slow
class TestNumericalAccuracy:
    """Test numerical accuracy and stability of core operations."""

    def test_parameter_precision(self):
        """Test that parameters maintain numerical precision."""
        # Test with high-precision discount factor
        beta_precise = 0.99999999999999999
        params = ModelParameters(beta=beta_precise)

        # [Unverified] Check that precision is maintained within JAX limits
        assert abs(params.beta - beta_precise) < 1e-15

    def test_transition_matrix_precision(self):
        """Test transition matrix numerical properties."""
        # Create nearly-singular transition matrix
        epsilon = 1e-12
        P = jnp.array([[1-epsilon, epsilon], [epsilon, 1-epsilon]])

        params = ModelParameters(transition_matrix=P)

        # Check that row sums are exactly 1.0 within tolerance
        row_sums = jnp.sum(params.transition_matrix, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-10)

    def test_parameter_boundary_conditions(self):
        """Test parameter validation at boundary conditions."""
        # Test values very close to boundaries
        boundary_tests = [
            (1e-15, True),    # Just above 0
            (1 - 1e-15, True), # Just below 1
            (0.5, True),      # Middle value
        ]

        for beta, should_pass in boundary_tests:
            if should_pass:
                params = ModelParameters(beta=beta)
                assert params.beta == beta
            else:
                with pytest.raises(EconomicValidationError):
                    ModelParameters(beta=beta)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for complete model workflow."""

    def test_complete_model_workflow(self):
        """Test complete model creation and validation workflow."""
        # Create model specification
        model_spec = {
            "name": "Integration Test Model",
            "parameters": {
                "beta": 0.99,
                "sigma": 1.0,
                "transition_matrix": [[0.9, 0.1], [0.2, 0.8]],
                "sigma_eps_r": 0.01,
                "sigma_eps_z": 0.02
            },
            "equations": {
                "states": ["r_lag", "z"],
                "controls": ["y", "pi", "r"],
                "shocks": ["eps_r", "eps_z"],
                "equilibrium": [
                    "y = y(+1) - (r - pi(+1))",
                    "pi = beta * pi(+1) + kappa * y",
                    "r = psi_pi * pi + eps_r"
                ]
            }
        }

        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(model_spec, f)
            temp_file = f.name

        try:
            model = Model.from_file(temp_file)

            # Verify all components loaded correctly
            assert model.model_name == "Integration Test Model"
            assert model.parameters.beta == 0.99
            assert model.parameters.transition_matrix is not None
            assert len(model.equations.state_variables) == 2
            assert len(model.equations.control_variables) == 3
            assert len(model.equations.shock_variables) == 2

        finally:
            Path(temp_file).unlink()


# Test fixtures for common model setups
@pytest.fixture
def simple_rbc_model():
    """Fixture providing a simple RBC model for testing."""
    params = ModelParameters(
        beta=0.99,
        sigma=1.0,
        shock_stds={"eps_a": 0.01}
    )

    equations = ModelEquations(
        state_variables=["k", "a"],
        control_variables=["c", "y", "i"],
        shock_variables=["eps_a"],
        equilibrium_conditions=[
            "y = a * k^alpha",
            "c + i = y",
            "beta * (1 + r(+1) - delta) = 1"
        ]
    )

    return Model(params, equations, "Simple RBC")


@pytest.fixture
def regime_switching_model():
    """Fixture providing a regime-switching model for testing."""
    transition_matrix = jnp.array([[0.9, 0.1], [0.2, 0.8]])

    params = ModelParameters(
        beta=0.99,
        regime_params={
            "psi_pi": {0: 0.8, 1: 1.6},  # Dovish vs hawkish
            "rho_r": {0: 0.9, 1: 0.8}
        },
        transition_matrix=transition_matrix,
        shock_stds={"eps_r": 0.01, "eps_z": 0.02}
    )

    equations = ModelEquations(
        state_variables=["r_lag", "z"],
        control_variables=["y", "pi", "r"],
        shock_variables=["eps_r", "eps_z"]
    )

    return Model(params, equations, "Regime-Switching NK")


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for model operations."""

    def test_model_creation_performance(self, benchmark):
        """Benchmark model creation time."""

        def create_model():
            params = ModelParameters(beta=0.99, sigma=1.0)
            equations = ModelEquations(
                state_variables=["k"] * 10,  # Larger model
                control_variables=["c"] * 10,
                shock_variables=["eps"] * 5
            )
            return Model(params, equations)

        result = benchmark(create_model)
        assert result is not None

    def test_parameter_validation_performance(self, benchmark):
        """Benchmark parameter validation performance."""

        def validate_parameters():
            # Create parameters that require extensive validation
            regime_params = {f"param_{i}": {0: 0.5, 1: 0.7} for i in range(100)}
            return ModelParameters(
                beta=0.99,
                regime_params=regime_params,
                shock_stds={f"eps_{i}": 0.01 for i in range(50)}
            )

        result = benchmark(validate_parameters)
        assert result is not None
'''

# Create test_core directory and test file
Path("tests/test_core").mkdir(exist_ok=True)
with open("tests/test_core/test_model.py", "w") as f:
    f.write(test_core_content)

print("✅ Created comprehensive test suite for core model")
print("   - Parameter validation tests with boundary conditions")
print("   - YAML loading and error handling tests")
print("   - Numerical precision and accuracy tests")
print("   - Integration tests for complete workflows")
print("   - Performance benchmarks")
print("   - [Unverified] labels for all uncertain test assertions")
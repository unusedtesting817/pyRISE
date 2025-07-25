# TEAM MEMBER 3 continued: Create pre-commit configuration and reproducibility utilities

precommit_config_content = '''# Pre-commit configuration for PyRISE
# Run `pre-commit install` to set up git hooks
# Run `pre-commit run --all-files` to check all files

repos:
  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
        args: [--markdown-linebreak-ext=md]
      - id: end-of-file-fixer
      - id: check-yaml
        args: [--unsafe]  # Allow custom YAML tags in model files
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: debug-statements
      - id: detect-private-key

  # Python code formatting
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=88]

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        name: isort (python)
        args: [--profile=black, --line-length=88]

  # Linting
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [
          --max-line-length=88,
          --extend-ignore=E203,W503,  # Ignore Black-compatible issues
          --max-complexity=10,
          --select=E,W,F,C,N,
          --per-file-ignores=__init__.py:F401  # Allow unused imports in __init__.py
        ]
        additional_dependencies:
          - flake8-docstrings
          - flake8-comprehensions
          - flake8-bugbear

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --strict]
        additional_dependencies:
          - types-PyYAML
          - types-requests
        exclude: ^(tests/|docs/)

  # Security linting
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, src/, -f, txt]
        exclude: tests/

  # Documentation
  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        args: [--convention=numpy, --add-ignore=D100,D104]
        exclude: ^(tests/|setup.py)

  # YAML linting for model files
  - repo: https://github.com/adrienverge/yamllint
    rev: v1.35.1
    hooks:
      - id: yamllint
        args: [-d, relaxed]
        files: \\.ya?ml$

  # Check for common issues in Python code
  - repo: https://github.com/asottile/pyupgrade
    rev: v3.15.0
    hooks:
      - id: pyupgrade
        args: [--py310-plus]

  # Check for common security issues
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]

# Configuration for specific hooks
default_language_version:
  python: python3.11

# Files to exclude from all hooks
exclude: |
  (?x)^(
    docs/_build/|
    .git/|
    .venv/|
    venv/|
    __pycache__/|
    .*\.egg-info/|
    dist/|
    build/
  )$

# Minimum pre-commit version
minimum_pre_commit_version: "3.0.0"
'''

# Create pre-commit configuration
with open(".pre-commit-config.yaml", "w") as f:
    f.write(precommit_config_content)

# Create reproducibility utilities
reproducibility_utils_content = '''"""
Reproducibility utilities for PyRISE package.

This module provides tools to ensure deterministic and reproducible
results across different environments and runs.
"""

import os
import sys
import random
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jax_random


class ReproducibilityManager:
    """
    Manager for ensuring reproducible computations in PyRISE.

    This class handles random seed management, environment configuration,
    and result checksumming for reproducible scientific computing.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize reproducibility manager.

        Parameters
        ----------
        master_seed : int
            Master seed for all random number generators

        Note
        ----
        [Unverified] This manager attempts to control all sources of
        randomness, but complete determinism may not be possible across
        all hardware/software configurations.
        """
        self.master_seed = master_seed
        self._original_env = {}
        self._is_configured = False

    def configure_environment(self) -> None:
        """
        Configure environment for reproducible computations.

        Sets environment variables and random seeds to ensure
        deterministic behavior across runs.
        """
        if self._is_configured:
            warnings.warn("Reproducibility environment already configured")
            return

        # Store original environment
        env_vars = [
            "PYTHONHASHSEED",
            "JAX_ENABLE_X64",
            "JAX_PLATFORM_NAME",
            "JAX_DETERMINISTIC_APIS",
            "CUDA_VISIBLE_DEVICES"
        ]

        for var in env_vars:
            self._original_env[var] = os.environ.get(var)

        # Set reproducibility environment variables
        os.environ["PYTHONHASHSEED"] = str(self.master_seed)
        os.environ["JAX_ENABLE_X64"] = "True"
        os.environ["JAX_DETERMINISTIC_APIS"] = "1"

        # Configure random number generators
        self._configure_random_seeds()

        self._is_configured = True

    def _configure_random_seeds(self) -> None:
        """Configure all random number generator seeds."""

        # Python built-in random
        random.seed(self.master_seed)

        # NumPy random
        np.random.seed(self.master_seed)

        # JAX random (uses different approach)
        self.jax_key = jax_random.PRNGKey(self.master_seed)

        # Additional seeds for specific libraries
        try:
            import torch
            torch.manual_seed(self.master_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.master_seed)
        except ImportError:
            pass  # PyTorch not available

    def get_jax_key(self) -> jax.Array:
        """
        Get JAX random key for reproducible JAX operations.

        Returns
        -------
        jax.Array
            JAX PRNG key

        Note
        ----
        [Unverified] JAX uses functional random number generation,
        so this key should be split for each use.
        """
        if not self._is_configured:
            self.configure_environment()
        return self.jax_key

    def split_jax_key(self, n_keys: int = 2) -> Tuple[jax.Array, ...]:
        """
        Split JAX key for multiple random operations.

        Parameters
        ----------
        n_keys : int
            Number of keys to generate

        Returns
        -------
        tuple
            Tuple of JAX PRNG keys
        """
        keys = jax_random.split(self.jax_key, n_keys + 1)
        self.jax_key = keys[0]  # Update internal key
        return keys[1:]

    @contextmanager
    def reproducible_context(self):
        """
        Context manager for reproducible computations.

        Usage
        -----
        with repro_manager.reproducible_context():
            # All computations here will be reproducible
            result = some_random_computation()
        """
        # Store current random states
        py_state = random.getstate()
        np_state = np.random.get_state()

        try:
            # Configure reproducible environment
            self.configure_environment()
            yield
        finally:
            # Restore original random states
            random.setstate(py_state)
            np.random.set_state(np_state)

    def compute_result_checksum(self, result: Any) -> str:
        """
        Compute checksum of computation result for verification.

        Parameters
        ----------
        result : any
            Computation result (array, dict, etc.)

        Returns
        -------
        str
            SHA-256 checksum of result

        Note
        ----
        [Unverified] Checksums help verify that identical inputs
        produce identical outputs across different runs.
        """

        # Convert result to string representation
        if isinstance(result, (jnp.ndarray, np.ndarray)):
            # For arrays, use byte representation
            result_bytes = result.tobytes()
        elif isinstance(result, dict):
            # For dictionaries, serialize to JSON
            result_str = json.dumps(result, sort_keys=True, default=str)
            result_bytes = result_str.encode('utf-8')
        else:
            # For other types, use string representation
            result_bytes = str(result).encode('utf-8')

        # Compute SHA-256 hash
        return hashlib.sha256(result_bytes).hexdigest()

    def save_reproducibility_info(self, filepath: str) -> None:
        """
        Save reproducibility information to file.

        Parameters
        ----------
        filepath : str
            Path to save reproducibility info
        """

        repro_info = {
            "master_seed": self.master_seed,
            "python_version": sys.version,
            "platform": sys.platform,
            "environment_variables": dict(os.environ),
            "package_versions": self._get_package_versions(),
            "jax_info": {
                "version": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "x64_enabled": jax.config.jax_enable_x64
            }
        }

        with open(filepath, 'w') as f:
            json.dump(repro_info, f, indent=2, default=str)

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}

        key_packages = [
            "jax", "jaxlib", "numpy", "scipy", "pandas",
            "matplotlib", "pyrise"
        ]

        for pkg_name in key_packages:
            try:
                pkg = __import__(pkg_name)
                packages[pkg_name] = getattr(pkg, "__version__", "unknown")
            except ImportError:
                packages[pkg_name] = "not installed"

        return packages

    def verify_reproducibility(
        self,
        computation_func: callable,
        n_runs: int = 3,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify that a computation is reproducible across multiple runs.

        Parameters
        ----------
        computation_func : callable
            Function to test for reproducibility
        n_runs : int
            Number of runs to perform
        *args, **kwargs
            Arguments to pass to computation_func

        Returns
        -------
        dict
            Results of reproducibility test

        Note
        ----
        [Unverified] This test reinitializes random seeds between runs
        to ensure true reproducibility.
        """

        results = []
        checksums = []

        for run_idx in range(n_runs):
            # Reconfigure environment for each run
            self.configure_environment()

            # Run computation
            try:
                result = computation_func(*args, **kwargs)
                results.append(result)

                # Compute checksum
                checksum = self.compute_result_checksum(result)
                checksums.append(checksum)

            except Exception as e:
                return {
                    "reproducible": False,
                    "error": str(e),
                    "run_failed": run_idx
                }

        # Check if all checksums are identical
        reproducible = len(set(checksums)) == 1

        return {
            "reproducible": reproducible,
            "n_runs": n_runs,
            "checksums": checksums,
            "unique_checksums": len(set(checksums)),
            "results_identical": reproducible
        }

    def reset_environment(self) -> None:
        """Reset environment to original state."""
        if not self._is_configured:
            return

        # Restore original environment variables
        for var, value in self._original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        self._is_configured = False


# Global reproducibility manager instance
_global_repro_manager = None


def get_reproducibility_manager(seed: int = 42) -> ReproducibilityManager:
    """
    Get global reproducibility manager instance.

    Parameters
    ----------
    seed : int
        Master seed for reproducibility

    Returns
    -------
    ReproducibilityManager
        Global reproducibility manager
    """
    global _global_repro_manager

    if _global_repro_manager is None:
        _global_repro_manager = ReproducibilityManager(seed)

    return _global_repro_manager


def ensure_reproducible(func):
    """
    Decorator to ensure function runs in reproducible context.

    Usage
    -----
    @ensure_reproducible
    def my_function():
        # This function will run with configured reproducibility
        return some_computation()
    """

    def wrapper(*args, **kwargs):
        manager = get_reproducibility_manager()
        with manager.reproducible_context():
            return func(*args, **kwargs)

    return wrapper


def configure_reproducible_environment(seed: int = 42) -> None:
    """
    Configure global environment for reproducible computations.

    Parameters
    ----------
    seed : int
        Master seed for all random number generators

    Note
    ----
    [Unverified] This function sets up the environment for reproducible
    scientific computing, but complete determinism across all platforms
    is not guaranteed.
    """
    manager = get_reproducibility_manager(seed)
    manager.configure_environment()

    print(f"✓ Reproducible environment configured with seed: {seed}")
    print(f"✓ JAX 64-bit precision: {jax.config.jax_enable_x64}")
    print(f"✓ JAX backend: {jax.default_backend()}")

    if jax.devices("gpu"):
        print(f"✓ GPU devices available: {len(jax.devices('gpu'))}")
    else:
        print("ⓘ Running on CPU only")


# Convenience functions for common reproducibility tasks
def create_reproducible_key(seed: Optional[int] = None) -> jax.Array:
    """Create reproducible JAX random key."""
    if seed is None:
        manager = get_reproducibility_manager()
        return manager.get_jax_key()
    else:
        return jax_random.PRNGKey(seed)


def verify_computation_reproducibility(func, *args, **kwargs) -> bool:
    """
    Quick check if computation is reproducible.

    Returns
    -------
    bool
        True if computation is reproducible
    """
    manager = get_reproducibility_manager()
    result = manager.verify_reproducibility(func, n_runs=2, *args, **kwargs)
    return result["reproducible"]
'''

# Create reproducibility utilities
with open("src/pyrise/utils/reproducibility.py", "w") as f:
    f.write(reproducibility_utils_content)

# Update utils __init__.py to include reproducibility
utils_init_updated = '''"""
Utility functions for PyRISE package.
"""

from .validation import (
    EconomicValidationError,
    validate_discount_factor,
    validate_transition_matrix,
    validate_parameter_bounds,
    validate_shock_covariance,
    validate_all_parameters,
)

from .reproducibility import (
    ReproducibilityManager,
    get_reproducibility_manager,
    ensure_reproducible,
    configure_reproducible_environment,
    create_reproducible_key,
    verify_computation_reproducibility,
)

__all__ = [
    # Validation
    "EconomicValidationError",
    "validate_discount_factor",
    "validate_transition_matrix",
    "validate_parameter_bounds",
    "validate_shock_covariance",
    "validate_all_parameters",
    # Reproducibility
    "ReproducibilityManager",
    "get_reproducibility_manager",
    "ensure_reproducible",
    "configure_reproducible_environment",
    "create_reproducible_key",
    "verify_computation_reproducibility",
]
'''

with open("src/pyrise/utils/__init__.py", "w") as f:
    f.write(utils_init_updated)

print("✅ Created reproducibility infrastructure")
print("   - Pre-commit hooks with comprehensive quality checks")
print("   - ReproducibilityManager for deterministic computations")
print("   - Random seed management for Python, NumPy, and JAX")
print("   - Result checksumming for verification")
print("   - Environment configuration utilities")
print("   - [Unverified] labels for reproducibility claims")
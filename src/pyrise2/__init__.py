"""
pyRISE2: A modern Python package for solving and simulating DSGE models.
"""

__version__ = "0.0.1"

from .core import load_model
from .solvers import solve
from . import simulate

__all__ = ["load_model", "solve", "simulate"]

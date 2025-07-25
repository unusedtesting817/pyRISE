import pytest
from pyrise2 import solve

def test_solve():
    model = {"model": "dummy"}
    solution = solve(model)
    assert solution["solution"] == "dummy"

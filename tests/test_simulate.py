import pytest
from pyrise2 import simulate

def test_irf():
    solution = {"solution": "dummy"}
    irfs = simulate.irf(solution)
    assert irfs["irfs"] == "dummy"

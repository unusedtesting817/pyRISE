import pytest
import sympy
from pyrise2.utils import differentiate

def test_differentiate_scalar():
    x = sympy.Symbol('x')
    f = x**2
    dfdx = differentiate([f], [x])
    assert dfdx[0, 0] == 2*x

def test_differentiate_vector():
    x, y = sympy.symbols('x y')
    f = [x**2 + y, y**2]
    df = differentiate(f, [x, y])
    assert df[0, 0] == 2*x
    assert df[0, 1] == 1
    assert df[1, 0] == 0
    assert df[1, 1] == 2*y

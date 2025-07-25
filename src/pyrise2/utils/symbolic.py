import sympy

def differentiate(equations, variables):
    """
    Differentiate a set of equations with respect to a set of variables.

    Args:
        equations (list): A list of sympy expressions.
        variables (list): A list of sympy symbols.

    Returns:
        sympy.Matrix: The Jacobian matrix of the system.
    """
    return sympy.Matrix(equations).jacobian(variables)

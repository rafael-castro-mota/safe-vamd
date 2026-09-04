import numpy as np
from typing import Sequence

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


def glq_quadrature_new(funcs: Sequence[MathFunc], funcs2: Sequence[MathFunc], inter: np.ndarray) -> float | complex:
    """
    This functions two lists of mathematical functions (MathFunc) and performs a Gauss Legendre Quadrature
    over the desired interval [inter[0], inter[1]] on the product of the mathematical functions (f1*f2*f3*...).

    Parameters
    ----------
    funcs: Sequence[MathFunc]
       First list of mathematical functions (MathFunc).
    funcs2: Sequence[MathFunc]
       Second list of mathematical functions (MathFunc).
    inter: np.ndarray of shape (2,1)
       Interpolation interval
    Returns
    -------
     float or complex
        Value of the integration in the desired interval.
    """

    glq_weights = [0.555555555555555, 0.888888888888888, 0.555555555555555]
    glq_points = [-0.77459666924148, 0,  0.77459666924148]

    # Converting Gauss Legendre points [-1,1] into the interval
    glq_points = (abs(inter[1] - inter[0]) / 2) * np.array(glq_points) + (inter[1] + inter[0]) / 2

    # Executing the Gauss Legendre Quadrature

    values = [np.prod([fun.value(x) for fun in funcs]) * np.prod([func.value(x) for func in funcs2]) for x in glq_points]

    value = np.sum(np.multiply(glq_weights, values))

    # Using the integral property of linearly mapped coordinates
    value = value * abs(inter[1] - inter[0]) / 2

    return value

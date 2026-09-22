from typing import Sequence
import numpy as np
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


def glq_quadrature(funcs: Sequence[MathFunc], inter: np.ndarray) -> float | complex:

    """
    This functions a list of mathematical functions (MathFunc) and performs a Gauss Legendre Quadrature
    over the desired interval [inter[0], inter[1]] on the product of the mathematical functions (f1*f2*f3*...).

    Parameters
    ----------
    funcs: Sequence[MathFunc]
        List of mathematical functions (MathFunc).
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
    for p in range(0, len(glq_points)):
        glq_points[p] = (abs(inter[1] - inter[0]) / 2) * glq_points[p] + (inter[1] + inter[0]) / 2

    # Executing the Gauss Legendre Quadrature
    value = 0
    for point in range(0, len(glq_points)):
        x = glq_points[point]
        point_value = 1
        for j in range(0, len(funcs)):
            point_value = point_value * funcs[j].value(x)

        value = value + glq_weights[point] * point_value

    # Using the integral property of linearly mapped coordinates
    value = value * abs(inter[1] - inter[0]) / 2
    return value

import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class PolyFunc(MathFunc):
    """
    A polynomial function implemented as a subclass of MathFunc.

    Attributes
    ----------
    coeffs: np.ndarray
        The coefficients of the polynomial function (f(x) = coeffs[0] + coeffs[1] * x + coeffs[2] * x^2 + ...)
    """
    def __init__(self, coeffs: np.ndarray) -> None:
        """
        Initializes an instance for PolyFunc.
        """

        self.coeffs = coeffs

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the polynomial function at x

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value of the polynomial function at x.
        """
        x = np.asarray(x)
        val = 0
        for i in range(0, len(self.coeffs)):
            val = val + self.coeffs[i] * (x ** i)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the polynomial function at x

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value of the derivative of the polynomial function at x.
        """

        val = 0
        x = np.asarray(x)
        for i in range(1, len(self.coeffs)):
            val = val + i * self.coeffs[i] * (x ** (i - 1))
        return val

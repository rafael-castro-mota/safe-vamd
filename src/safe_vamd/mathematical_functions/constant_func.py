import numpy as np
from numbers import Complex
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class ConstFunc(MathFunc):
    """
    A constant function f(x) = Cst., implemented as a subclass of MathFunc.

     Attributes
    ----------
    const : Complex
        Value (real or imaginary) for the constant
    """

    def __init__(self, const: Complex) -> None:
        """
        initializes an instance of CompositeFunc.

        Parameters
        ----------
        const : Complex
            Complex-valued constant
        """

        self.const = const

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the constant function f(x) = Cst. at x(s).

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value(s) of the constant function at x(s). The value is equal to the constant everywhere.
        """
        x = np.asarray(x)
        return complex(self.const) * np.ones(np.shape(x))

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the constant function f(x) = Cst. at x(s).

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value(s) of the constant function at x(s). The derivative is zero everywhere.
        """
        return np.zeros(np.shape(x))

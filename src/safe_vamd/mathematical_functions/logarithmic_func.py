
import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class LogFunc(MathFunc):
    """
    A natural logarithm function f(x) = ln(f1(x)), implemented as a subclass of MathFunc.

     Attributes
    ----------
    f1 : MathFunc
        Function in the power.
    """
    def __init__(self, f1: MathFunc):
        """
        Initializes an instance of LogFunc.
        Parameters
        ----------
        f1
        """
        self.f1 = f1

    def value(self, x: ArrayLike) -> np.ndarray:

        """
        Evaluates the natural logarithm function f(x) = ln(f1(x)) at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the natural logarithm function at x(s).
        """

        val = np.log(self.f1.value(x))
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:

        """
        Evaluates the derivative of the natural logarithm function f(x) = ln(f1(x)) at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative of the natural logarithm function at x(s).
        """

        val = self.f1.dvalue(x) * (1 / self.f1.value(x))
        return val

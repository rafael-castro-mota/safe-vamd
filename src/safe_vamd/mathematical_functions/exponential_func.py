import math
import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc
import math


class ExpFunc(MathFunc):
    """
    A natural exponential function f(x) = e^(f1(x)), implemented as a subclass of MathFunc.

     Attributes
    ----------
    f1 : MathFunc
        Function in the exponent.
    """

    def __init__(self, f1: MathFunc) -> None:
        """ Initializes an instance of ExpFunc"""
        self.f1 = f1

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the natural exponential function e^(f1(x)) at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the natural exponential function at x(s).
        """
        val = math.e ** self.f1.value(x)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the natural exponential function e^(f1(x)) at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative of the natural exponential function at x(s).
        """

        val = self.f1.dvalue(x) * self.value(x)
        return val

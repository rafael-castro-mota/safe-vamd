import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class PowerFunc(MathFunc):
    """
    A power function where both the base and the exponent are mathematical functions f(x) = f1(x) ^ f2(x)
     implemented as a subclass of MathFunc.

     Attributes
    ----------
    f1 : MathFunc
        Mathematical function for the base.
    f2 : MathFunc
        Mathematical function for the exponent.
    """

    def __init__(self, f1: MathFunc, f2: MathFunc) -> None:
        """
        Initializes an instance of PowerFunc.
        """

        self.f1 = f1
        self.f2 = f2

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the power function at x(s).

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value(s) of the power function at x(s).
        """

        x = np.array(x)
        val = self.f1.value(x) ** self.f2.value(x)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the power function at x(s).

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Value(s) of the derivative of the power function at x(s).
        """
        x = np.asarray(x)
        val = ((self.f1.value(x)) ** self.f2.value(x)) * np.log(self.f1.value(x)) * self.f2.dvalue(x)
        return val


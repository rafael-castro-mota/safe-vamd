import numpy as np
from numpy.typing import ArrayLike
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class DivFunc(MathFunc):
    """
    The division between two functions f(x) = f1(x)/f2(x), implemented as a subclass of MathFunc.

    Attributes
    ----------
    f1 : MathFunc
        Function in the numerator.
    f2 : MathFunc
        Function in the denominator.
    """

    def __init__(self, f1, f2):
        """
        Initializes an instance of DivFunc.
        """

        self.f1 = f1
        self.f2 = f2

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the division between f1 and f2 at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the division of f1 with f2 at x(s).
        """

        val = self.f1.value(x) / self.f2.value(x)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of division between f1 and f2 at x(s).

        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative of the division of f1 with f2 at x(s).
        """

        val = self.f1.dvalue(x) * self.f2.value(x) - self.f1.value(x) * self.f2.dvalue(x)
        val = val / (self.f2.value(x) ** 2)
        return val


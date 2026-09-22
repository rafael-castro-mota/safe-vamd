import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class SumFunc(MathFunc):
    """
    The summation of two mathematical functions (f(x) = f1(x) + f2(x)), implemented as a subclass of MathFunc.

     Parameters
    ----------
    f1: MathFunc
        The first mathematical function
    f2: MathFunc
        The second mathematical function
    """
    def __init__(self, f1: MathFunc, f2: MathFunc) -> None:
        """
        Initializes an instance of SumFunc.
        """

        self.f1 = f1
        self.f2 = f2

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the sum of functions f1 and f2 at x(s)

        Parameters
        ----------
        x: ArrayLike
            input value(s)
        Returns
        -------
        np.ndarray
            Value(s) of the sum of functions f1 and f2 at x(s)
        """

        x = np.asarray(x)
        val = self.f1.value(x) + self.f2.value(x)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
         Evaluates the derivative of the sum of functions f1 and f2 at x(s)

         Parameters
         ----------
         x: ArrayLike
             input value(s)
         Returns
         -------
         np.ndarray
             Value(s) of derivative of the sum of functions f1 and f2 at x(s)
         """

        x = np.asarray(x)
        val = self.f1.dvalue(x) + self.f2.dvalue(x)
        return val

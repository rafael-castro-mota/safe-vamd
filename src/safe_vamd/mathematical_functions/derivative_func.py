import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class DerivFunc(MathFunc):
    """
    A derivative of a function f(x) = df1/dx, implemented as a subclass of MathFunc.

     Attributes
    ----------
    f1 : MathFunc
        Function for which the derivative is to be computed.
    """

    def __init__(self, f1) -> None:
        """
        Initializes an instance of DerivFunc.
        """

        self.f1 = f1

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Computes the derivative of f1 with respect to x by accessing its .dvalue method.
        Parameters
        ----------
        x: ArrayLike
            Input value(s).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to x of function f1 at x(s).
        """

        val = self.f1.dvalue(x)
        return val

    def dvalue(self, x):
        print("Second Derivatives are not currently supported!!!")
        return None

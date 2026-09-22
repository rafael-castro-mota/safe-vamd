
import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class CompositeFunc(MathFunc):
    """
    A composite function f(x) = f1(f2(x)), implemented as a subclass of MathFunc.

    This class implements function composition and its derivative using the chain rule.


    Attributes
    ----------
    f1 : MathFunc
        Outer function in the composition.
    f2 : MathFunc
        Inner function in the composition.
    """
    def __init__(self, f1: MathFunc, f2: MathFunc) -> None:
        """
        initializes an instance of CompositeFunc.
        """

        self.f1 = f1
        self.f2 = f2

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the composite function f(x) = f1(f2(x)) at x(s).

        Parameters
        ----------
        x : ArrayLike
            Input value(s).

        Returns
        -------
        np.ndarray
            Value(s) of the composite function at x(s).
        """
        x = np.asarray(x)
        val = self.f1.value(self.f2.value(x))
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the composite function's derivative at x(s).

        Parameters
        ----------
        x : ArrayLike
           Input value(s).

        Returns
        -------
        np.ndarray
            Value(s) of the composite function's derivative at x(s).
        """
        x = np.asarray(x)
        val = self.f1.dvalue(self.f2.value(x))*self.f2.dvalue(x)
        return val

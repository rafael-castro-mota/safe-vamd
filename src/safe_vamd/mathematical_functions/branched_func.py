
import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class BranchedFunc(MathFunc):
    """
    A branched function f(x), implemented as a subclass of MathFunc.

    f(x) = {f1(x), x < xchange; f2(x), x > xchange; 0.5 * (f1(x) + f2(x)), x=xchange}

     Attributes
    ----------
    f1 : MathFunc
        "Left" function.
    f2 : MathFunc
        "Right" function.
    """
    def __init__(self, f1: MathFunc, f2: MathFunc, xchange: float) -> None:
        """
        initializes an instance of BranchedFunc.

        Parameters
        ----------
        f1 : MathFunc
            "Left" function.
        f2 : MathFunc
            "Right" function.
        xchange : float
            Where the function switches between the left and right functions.
        """
        self.f1 = f1
        self.f2 = f2
        self.xchange = xchange

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the branched function at x(s).

        Parameters
        ----------
        x : ArrayLike
            Input value(s).

        Returns
        -------
        np.ndarray
            Value(s) of the branched function at x(s).
        """
        x = np.asarray(x)

        f1_vals = self.f1.value(x)
        f2_vals = self.f2.value(x)

        val = np.where(x < self.xchange, f1_vals, f2_vals)

        mask_mid = x == self.xchange
        if np.any(mask_mid):
            val = val.copy()
            val[mask_mid] = 0.5 * (f1_vals[mask_mid] + f2_vals[mask_mid])

        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the branched function's derivative at x(s).

        Parameters
        ----------
        x : ArrayLike
            Input value(s).

        Returns
        -------
        np.ndarray
            Value(s) of the branched function's derivative at x(s).
        """

        x = np.asarray(x)

        f1_vals = self.f1.dvalue(x)
        f2_vals = self.f2.dvalue(x)

        val = np.where(x < self.xchange, f1_vals, f2_vals)

        mask_mid = x == self.xchange
        if np.any(mask_mid):
            val = val.copy()
            val[mask_mid] = 0.5 * (f1_vals[mask_mid] + f2_vals[mask_mid])

        return val


import numpy as np
from typing import Sequence
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class ProdFunc(MathFunc):
    """
    The product of a list mathematical functions (f(x)= funcs[0](x) * funcs[2](x) * funcs[2](x) * ...),
    implemented as a subclass of MathClass

    Attributes
    ----------
    funcs: Sequence[MathFunc]
        List of mathematical functions.
    """
    def __init__(self, funcs: Sequence[MathFunc]) -> None:
        """
        Initializes an instance of ProdFunc.
        """

        self.funcs = funcs

    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Computes the product of the mathematical functions at x(s)

        Parameters
        ----------
        x: ArrayLike
            Input value(s)

        Returns
        -------
        np.ndarray
            Value(s) the product of the mathematical functions at x(s)
        """
        x = np.asarray(x)
        if x.ndim != 0:
            val = np.ones(len(x))
        else:
            val = 1

        for i in range(0, len(self.funcs)):
            val = val * self.funcs[i].value(x)
        return val

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Computes the derivative of the product of the mathematical functions at x(s)

        Parameters
        ----------
        x: ArrayLike
            Input value(s)

        Returns
        -------
        np.ndarray
            Value(s) ofr the derivative of the product of the mathematical functions at x(s)
        """
        x = np.asarray(x)
        if x.ndim != 0:
            val = np.ones(len(x))
        else:
            val = 1

        for i in range(0, len(self.funcs)):
            a = self.funcs[i].dvalue(x)
            for j in range(0, len(self.funcs)):
                if j != i:
                    a = a * self.funcs[j].value(x)
            val = val + a

        return val
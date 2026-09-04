import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class WaveNumber(MathFunc):
    """
    The wave number, implemented as a subclass of MathFunc.

    Attributes
    ----------
    c: MathFunc
        Speed of sound (in m/S).
    w: MathFunc
        Excitation Frequency (in rads/s).
    """

    def __init__(self, c: MathFunc, w: MathFunc) -> None:
        """
        Initializes an instance of WaveNumber.
        """
        self.c = c
        self.w = w

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the wave number at z(s)

        Parameters
        ----------
        z: ArrayLike
          Height(s) (in meters).
        Returns
        -------
        np.ndarray
          Value(s) of the wave number at z(s)
        """

        z = np.asarray(z)
        val = self.w.value(z) / self.c.value(z)     # Local Wave Number (Type: Float) [-]
        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the wave number at z(s)

        Parameters
        ----------
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the wave number at z(s)
        """

        z = np.asarray(z)
        c = self.c.value(z)
        dc = self.c.dvalue(z)
        w = self.w.value(z)
        dw = self.w.dvalue(z)
        val = dw * np.real(c) - w * np.real(dc)
        val = val / (np.real(c) ** 2)
        return val

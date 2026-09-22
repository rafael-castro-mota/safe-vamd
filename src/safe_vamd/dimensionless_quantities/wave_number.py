import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class WaveNumber(MathFunc):
    """
    The wave number, implemented as a subclass of *MathFunc*.

    Attributes
    ----------
    c: MathFunc
        Speed of sound field (in :math:`m`), implemented as a *MathFunc*.
    w: MathFunc
        Excitation Frequency (in :math:`rads^{-1}`), implemented as a *MathFunc*.
    """

    def __init__(self, c: MathFunc, w: MathFunc) -> None:
        """

        Parameters
        ----------
        c: MathFunc
            Speed of sound field (in :math:`m`), implemented as a *MathFunc*.
        w: MathFunc
            Excitation Frequency (in :math:`rads^{-1}`), implemented as a *MathFunc*.
        """
        self.c = c
        self.w = w

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the wave number at z(s).

        Parameters
        ----------
        z: ArrayLike
          Input height(s) (in :math:`m`).
        Returns
        -------
        np.ndarray
          Value(s) of the wave number at z(s).
        """

        z = np.asarray(z)
        val = self.w.value(z) / self.c.value(z)     # Local Wave Number (Type: Float) [-]
        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the wave number at z(s).

        Parameters
        ----------
        z: ArrayLike
            Input height(s) (in :math:`m`).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the wave number at z(s).
        """

        z = np.asarray(z)
        c = self.c.value(z)
        dc = self.c.dvalue(z)
        w = self.w.value(z)
        dw = self.w.dvalue(z)
        val = dw * np.real(c) - w * np.real(dc)
        val = val / (np.real(c) ** 2)
        return val

import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class MachNumber(MathFunc):
    """
    The mach number, implemented as a subclass of *MathFunc*.

    **Class attributes:**

    Attributes
    ----------
    c: MathFunc
        Speed of sound field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
    v. MathFunc
        Wind-Velocity field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
    """

    def __init__(self, c: MathFunc, v: MathFunc):
        """

        Parameters
        ----------
        c: MathFunc
            Speed of sound field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
        v. MathFunc
            Wind-Velocity field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
        """

        self.c = c
        self.v = v

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the mach number at z(s).

        Parameters
        ----------
        z: ArrayLike
            Input height(s) (in :math:`m`).
        Returns
        -------
        np.ndarray
            Value(s) of the mach number at z(s).
        """

        z = np.array(z)
        val = self.v.value(z) / np.real(self.c.value(z))
        return val

    def dvalue(self, z):
        """
        Evaluates the derivative with respect to height of the mach number at z(s).

        Parameters
        ----------
        z: ArrayLike
            Input height(s) (in :math:`m`).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the mach number at z(s).
        """

        z = np.asarray(z)
        v = self.v.value(z)
        dv = self.v.dvalue(z)
        c = np.real(self.c.value(z))
        dc = np.real(self.c.dvalue(z))
        val = (dv * c - v * dc)/(c ** 2)
        return val

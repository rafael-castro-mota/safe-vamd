import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class MachNumber(MathFunc):
    """
    The mach number, implemented as a subclass of MathFunc.

    Attributes
    ----------
    c: MathFunc
        Speed of sound (in m/s).
    v. MathFunc
        Wind Velocity (in m/S)
    """

    def __init__(self, c: MathFunc, v: MathFunc):
        """
        Initializes an instance of MachNumber.
        """

        self.c = c  # Speed of Sound Object (Type: MathFunc)[m/s]
        self.v = v  # Velocity Object (Type: MathFunc) [m/s]

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the mach number at z(s)

        Parameters
        ----------
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the mach number at z(s)
        """

        z = np.array(z)
        val = self.v.value(z) / np.real(self.c.value(z))    # Local Mach Number (Type: Float) [-]
        return val

    def dvalue(self, z):
        """
        Evaluates the derivative with respect to height of the mach number at z(s)

        Parameters
        ----------
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the mach number at z(s)
        """

        z = np.asarray(z)
        v = self.v.value(z)             # Local Velocity (Type: Float) [-]
        dv = self.v.dvalue(z)           # Local Derivative of Velocity (Type: Float) [-]
        c = np.real(self.c.value(z))    # Local Speed of Sound (Type: Float) [-]
        dc = np.real(self.c.dvalue(z))  # Local Derivative of Speed of Sound (Type: Float) [-]
        val = (dv * c - v * dc)/(c ** 2)
        return val

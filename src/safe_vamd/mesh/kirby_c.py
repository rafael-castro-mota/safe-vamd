import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


# Kirby Perfectly Matched Layer Speed of Sound
class KirbyC(MathFunc):
    """
    Function for the speed of sound with an added imaginary term for absorption in the Perfectly Matched Layer,
    implemented as a subclass of MathFunc.

    Attributes
    ----------
    c : MathFunc
        Function for the speed of sound, implemented as a subclass of MathFunc.
    alpha : float
        Damping coefficient with units of dB/wavelength, implemented as a subclass of MathFunc.

    Notes
    ----------
    The expression for the modified speed of sound is presented in Eq. (24) in https://doi.org/10.1121/10.0002912.
    """
    def __init__(self, c: MathFunc, alpha: MathFunc) -> None:
        """
        Initializes an instance of class KirbyC.

        Parameters
        ----------
         c : MathFunc
            Function for the speed of sound, implemented as a subclass of MathFunc.
        alpha : MathFunc
            Damping coefficient with units of dB/wavelength, implemented as a subclass of MathFunc.
        """

        self.c = c
        self.alpha = alpha

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates of the stretching function at z(s).

        Parameters
        ----------
        z : ArrayLike
            z-coordinate(s) (height, in meters). Needs to be convertible to a numpy array.

        Returns
        ----------
        np.ndarray
            Value(s) of the speed of sound with the added imaginary absorption term at z(s).
        """

        z = np.asarray(z)
        mod = 1/(1 - 1j * ((self.alpha.value(z)) / (40 * np.pi * np.log10(np.e))))
        val = self.c.value(z) * mod

        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the stretching function at z(s).

        Parameters
        ----------
        z : ArrayLike
            Height(s) (in meters).

        Returns
        ----------
        np.ndarray
            Value(s) of the derivative with respect to height of the speed of sound with the added imaginary absorption
            term at z(s).
        """

        z = np.asarray(z)
        mod = 1 / (1 - 1j * (self.alpha.value(z) / (40 * np.pi * np.log10(np.e))))
        val = mod * self.c.dvalue(z)
        return val

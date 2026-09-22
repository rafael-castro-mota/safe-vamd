import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


# Kirby Perfectly Matched Layer Speed of Sound
class KirbyC(MathFunc):
    r"""
    Modified speed of sound field :math:`\hat{c}`, implemented as a subclass of *MathFunc*,
    with an added absorption term (imaginary) for the Perfectly Matched Layer.

    **Class attributes:**

    Attributes
    ----------
    c : MathFunc
         Speed of sound field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
    alpha : MathFunc
        Damping coefficient (in dB/wavelength), implemented as a *MathFunc*.

    Notes
    ----------
    The expression for the modified speed of sound is presented in Eq. 24 in https://doi.org/10.1121/10.0002912 and
    takes the form:

    .. math::
        \hat{c} = c[1-i\alpha\div(40\pi\log_{10}(e))]^{-1}
    , where :math:`c` is the original speed of sound field and :math:`\alpha` is a damping coefficient
    (in dB/wavelength).

    """
    def __init__(self, c: MathFunc, alpha: MathFunc) -> None:
        """

        Parameters
        ----------
        c : MathFunc
            Speed of sound field (in :math:`ms^{-1}`), implemented as a *MathFunc*.
        alpha : MathFunc
            Damping coefficient with units of dB/wavelength, implemented as a subclass of MathFunc.
        """

        self.c = c
        self.alpha = alpha

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the modified speed of sound at z(s).

        Parameters
        ----------
        z : ArrayLike
            Input height(s) (in :math:´m`).

        Returns
        ----------
        np.ndarray
            Value(s) of the modified speed of sound at z(s).
        """

        z = np.asarray(z)
        mod = 1/(1 - 1j * ((self.alpha.value(z)) / (40 * np.pi * np.log10(np.e))))
        val = self.c.value(z) * mod

        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the modified speed of sound at z(s).

        Parameters
        ----------
        z : ArrayLike
            Input height(s) (in :math:´m`).

        Returns
        ----------
        np.ndarray
            Value(s) of the derivative with respect to height of the modified speed of sound at z(s).
        """

        z = np.asarray(z)
        mod = 1 / (1 - 1j * (self.alpha.value(z) / (40 * np.pi * np.log10(np.e))))
        val = mod * self.c.dvalue(z)
        return val

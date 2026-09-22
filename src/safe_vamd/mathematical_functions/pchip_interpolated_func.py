
import numpy as np
from typing import Sequence
from numpy.typing import ArrayLike
from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import PPoly


class PchipInterpolator1D(MathFunc):
    """
    A PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for the real and imaginary parts of a data set,
    implemented as a subclass of MathFunc.

    Attributes
    ----------
    self.f_real: PchipInterpolator
        Real part(s) of the dependent variable interpolated by a PCHIP.
    self.f_imag: PchipInterpolator
        Imaginary part(s) of the dependent variable interpolated by a PCHIP.
    self.df_real: PPoly
        Derivative of the real part of the dependent variable interpolated by a PCHIP.
    self.df_imag: PPoly
        Derivative of the imaginary part of the dependent variable interpolated by a PCHIP.


    """

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Initializes and instance of PchipInterpolator1D.

        Parameters
        ----------
        x: ArrayLike
            Values for the independent variable.
        y: ArrayLike
            Values for the dependent variable.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        self.f_real = PchipInterpolator(x, np.real(y))
        self.f_imag = PchipInterpolator(x, np.imag(y))
        self.df_real = self.f_real.derivative()
        self.df_imag = self.f_imag.derivative()

    def value(self, x: ArrayLike) -> np.ndarray:
        """"
        Evaluates the PCHIP for the imaginary and real parts at x(s) and creates a complex value.

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Complex-value(s) built from the PCHIP for the imaginary and real parts at x(s)
        """

        x = np.asarray(x)
        return self.f_real(x) + 1j * self.f_imag(x)

    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """"
        Evaluates the derivative of the PCHIP for the imaginary and real parts at x(s) and creates a complex value.

        Parameters
        ----------
        x : array_like
            Input value(s).

        Returns
        -------
        array_like
            Complex-value(s) built from the derivatives for the PCHIP for the imaginary and real parts at x(s)
        """
        x = np.asarray(x)
        return self.df_real(x) + 1j * self.df_imag(x)


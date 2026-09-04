import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence

from safe_vamd.mathematical_functions.pchip_interpolated_func import PchipInterpolator1D
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class Mode(MathFunc):
    """
    A class representing an individual vertical atmospheric (VA) mode.

    Attributes
    ----------
    amplitude: float
        Modal amplitude.
    z_coords: Sequence[float]
        Heights (in meters) of the modal shape (eigenvector)
    eigval: complex
        Eigenvalue of the mode.
    eigvec: Sequence[complex]
        Mode shape (eigenvector) of the mode.
    eigvec_f: PchipInterpolator1D
        A Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) for the eigenvector
        that includes the imaginary and real parts.
    ref_wn: complex
        Reference wave-number.
    """
    def __init__(self, amplitude: float, z_coords: Sequence[float], eigval: complex,
                 eigvec: Sequence[complex], ref_wn: complex):
        """
        Initializes an instance of Mode
        """

        self.amplitude = amplitude
        self.z_coords = z_coords
        self.eigvec = eigvec
        self.eigvec_f = self.update_cubicspline_eig()
        self.eigval = eigval
        self.ref_wn = ref_wn

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the modal shape at z(s).

        Parameters
        ----------
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the modal shape at z(s).
        """

        z = np.asarray(z)
        return self.eigvec_f.value(z)

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the modal shape with respect to height at z(s).

         Parameters
         ----------
         z: ArrayLike
             Height(s) (in meters).
         Returns
         -------
         np.ndarray
             Value(s) of derivative with respect to height of the modal shape at z(s).
         """

        z = np.asarray(z)
        return self.eigvec_f.dvalue(z)

    def value_2d(self, x: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the mode at (x, z).

        Parameters
        ----------
        x: ArrayLike
            Range(s) (in meters).
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the mode at (x, z).
        """
        x = np.asarray(x)
        z = np.asarray(z)
        kx = self.ref_wn * self.eigval
        value = self.value(z) * np.exp(-1j * kx * x)
        return value

    def dvalue_x(self, x: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to range of the mode at (x, z)

        Parameters
        ----------
        x: ArrayLike
            Range(s) (in meters).
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to range of the mode at (x, z).
        """

        x = np.asarray(x)
        z = np.asarray(z)

        # Derivative of mode in x
        kx = self.ref_wn * self.eigval
        value = - 1j * kx * self.value_2d(x, z)
        return value

    def dvalue_z(self, x: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the mode at (x, z)

        Parameters
        ----------
        x: ArrayLike
            Range(s) (in meters).
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the mode at (x, z).
        """
        kx = self.ref_wn * self.eigval
        return self.dvalue(z) * np.exp(-1j * kx * x)

    def update_cubicspline_eig(self) -> PchipInterpolator1D:

        """
        Interpolates the real and imaginary parts of the mode eigenvector with a PchipInterpolator1D.
        """

        f = PchipInterpolator1D(self.z_coords, self.eigvec)
        self.eigvec_f = f
        return f

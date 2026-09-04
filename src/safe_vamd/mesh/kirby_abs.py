import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class KirbyABS(MathFunc):
    """
    Stretching function for the Perfectly Matched Layer (PML), implemented as a subclass of MathFunc.

    Attributes
    ----------
    tau1 : float
        Factor for the stretching function.
    tau2 : float
        Factor for the stretching function.
    z_ref : float
        z-coordinate (height, in meters) of the physical domain/PML interface.
    pml_thickness : float
        Thickness (in meters) of the Perfectly Matched Layer.

    Notes
    -----
    The stretching function can be found in Ray Kirby's journal article
    (Eq. (23) in https://doi.org/10.1121/10.0002912).
    """

    def __init__(self, tau1: float, tau2: float, z_ref: float, pml_thickness: float):
        """
        initializes an instance of class KirbyABS.

        Parameters
        ----------
        tau1 : float
            Factor for the stretching function.
        tau2 : float
            Factor for the stretching function.
        z_ref : float
            z-coordinate (height, in meters) of the physical domain/PML interface.
        pml_thickness : float
            Thickness (in meters) of the Perfectly Matched Layer.
        """

        self.tau1 = tau1
        self.tau2 = tau2
        self.z_ref = z_ref
        self.pml_thickness = pml_thickness

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the stretching function at.

        Parameters
        ----------
        z : Arraylike
            Height(s) (in meters).

        Returns
        ----------
        np.ndarray
            Value(s) of the stretching function at z(s).
        """

        z = np.asarray(z)
        dists = (z - self.z_ref) / self.pml_thickness
        vals1 = np.exp(dists * self.tau1) - 1j * (np.exp(dists * self.tau2) - 1)
        vals2 = 1
        val = np.where(dists > 0, vals1, vals2)

        return val

    def dvalue(self, z: ArrayLike) -> None:
        """
        No derivative is actually implemented. Only done to respect parent's class abstract method.
        """

        return None
import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class KirbyABS(MathFunc):
    r"""
    A stretching function :math:`\xi` for the Perfectly Matched Layer (PML), implemented as subclass of *MathFunc*.

    **Class attributes:**

    Attributes
    ----------
    tau1 : float
        Unitless factor for the stretching function.
    tau2 : float
        Unitless factor for the stretching function.
    z_ref : float
        z-coordinate (height, in :math:`m`) of the physical domain/PML interface.
    pml_thickness : float
        Thickness (in :math:`m`) of the Perfectly Matched Layer.

    Notes
    -----
    The stretching function :math:`\xi` is defined by Eq. 23 in https://doi.org/10.1121/10.0002912
    and takes the form:

    .. math::
        \xi=e^{\tau_1\overline{z}}-i[e^{\tau_2\overline{z}}-1]
    , where :math:`\overline{z}=(z-a)\div h` with :math:`a` being the inner domain's height (in :math:`m`) and :math:`h`
    the thickness (in :math:`m`) of the Perfectly Matched Layer.
    """

    def __init__(self, tau1: float, tau2: float, z_ref: float, pml_thickness: float):
        """

        Parameters
        ----------
        tau1 : float
            Unitless factor for the stretching function.
        tau2 : float
            Unitless factor for the stretching function.
        z_ref : float
            z-coordinate (height, in :math:`m`) of the physical domain/PML interface.
        pml_thickness : float
            Thickness (in :math:`m`) of the Perfectly Matched Layer.
        """

        self.tau1 = tau1
        self.tau2 = tau2
        self.z_ref = z_ref
        self.pml_thickness = pml_thickness

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the stretching function at z(s).

        Parameters
        ----------
        z : Arraylike
            Input height(s) (in :math:`m`).

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

        Notes
        -----
            No derivative is actually implemented. Only done to respect parent's class abstract method.
        """

        return None

import numpy as np
from numpy.typing import NDArray

from safe_vamd.source.source import Source
from safe_vamd.background_atmospheric_field.background_field import BackgroundField


class GroundIngardMyersBC:
    """
    A class that applies the Ingard-Myers boundary condition on the ground plane directly to
    the cubic eigenvalue problem matrices (:math:`[A+Bλ+Cλ^2+Cλ^3]p=0`) in the SAFE method.

    **Class attributes:**

    Attributes
    ----------
    source: Source
        An instance of *Source* with information about the acoustic source.
    background_field: BackgroundField
        An instance of *BackgroundField* representing the atmospheric background field.
    impedance: complex
        The complex-valued normalized surface ground impedance

    Notes
    -----
        The derivation of the Ingard-Myers impedance for the SAFE method is available in
        https://doi.org/10.1121/10.0003567.
    """
    def __init__(self, source: Source, background_field: BackgroundField, impedance: complex) -> None:
        """

        Parameters
        ----------
        source: Source
            An instance of *Source* with information about the acoustic source.
        background_field: BackgroundField
            An instance of *BackgroundField* representing the atmospheric background field.
        impedance: complex
            The complex-valued normalized surface ground impedance.
        """

        self.source = source
        self.background_field = background_field    # Background Flow Field
        self.impedance = impedance

    def apply_bc(self, a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> None:
        """
        Takes the matrices for the cubic eigenvalue problem :math:`[A+Bλ+Cλ^2+Cλ^3]p=0`
        and directly inserts the contributions of the Ingard-Myers impedance boundary condition
        into the first diagonal entry (ground).

        Parameters
        ----------
        a: NDArray
            Matrix :math:`A` in the cubic eigenvalue problem (:math:`A+Bλ+Cλ^2+Cλ^3`).
        b: NDArray
            Matrix :math:`B` in the cubic eigenvalue problem (:math:`A+Bλ+Cλ^2+Cλ^3`).
        c: NDArray
            Matrix :math:`C` in the cubic eigenvalue problem (:math:`A+Bλ+Cλ^2+Cλ^3`).
        d: NDArray
            Matrix :math:`D` in the cubic eigenvalue problem (:math:`A+Bλ+Cλ^2+Cλ^3`).

        Returns
        -------
        None
        """

        w = self.source.w
        u0 = self.background_field.vx.value(0)
        c0 = self.background_field.c.value(0)
        mach0 = abs(u0/np.real(c0))
        g0 = self.background_field.g.value(0)/(c0 ** 2)
        k0 = w/c0
        z = self.impedance

        a[0][0] = a[0][0] + ((1j*k0)/z)-g0
        b[0][0] = b[0][0] - mach0 * (((3*1j*k0)/z)-g0)
        c[0][0] = c[0][0] + (mach0 ** 2) * (3*1j*k0)/z
        d[0][0] = d[0][0] - (mach0 ** 3) * (1j*k0)/z

        return None

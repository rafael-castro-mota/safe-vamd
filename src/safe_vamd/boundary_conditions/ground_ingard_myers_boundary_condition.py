
import numpy as np
from numpy.typing import NDArray

from safe_vamd.source.source import Source
from safe_vamd.background_atmospheric_field.background_field import BackgroundField


class GroundIngardMyersBC:
    """
    A class that applies the Ingard-Myers boundary condition in the ground plane.

    Attributes
    ----------
    source: Source
        Information about the source held in an instance of the Source class.
    background_field: BackgroundField
        The fields for the atmospheric background quantities held in an instance of the Ba class.
    impedance: complex
    Complex-value ground normalized surface impedance.
    """
    def __init__(self, source: Source, background_field: BackgroundField, impedance: complex) -> None:
        """
        Initializes an instance of the GroundIngardMyersBC class.
        """

        self.source = source
        self.background_field = background_field    # Background Flow Field
        self.impedance = impedance

    def apply_bc(self, a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> None:
        """
        Takes the matrices for the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3)p=0
        and inserts the contributions of the IngardMyers impedance boundary condition to the first entry (ground) in
        each.

        Parameters
        ----------
        a: NDArray
        Matrix A in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
        b: NDArray
        Matrix B in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
        c: NDArray
        Matrix C in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).
        d: NDArray
        Matrix D in the cubic eigenvalue problem (A+Bλ+Cλ^2+Cλ^3).

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

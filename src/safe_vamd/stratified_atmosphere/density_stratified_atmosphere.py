import numpy as np
from typing import Sequence
from scipy.integrate import quad
from numpy.typing import ArrayLike

from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from safe_vamd.mathematical_functions.power_func import PowerFunc
from safe_vamd.mathematical_functions.constant_func import ConstFunc
from safe_vamd.background_atmospheric_field.air_properties import AirProperties


class RhoStratAtmo(MathFunc):
    """
    A class that computes the air density in hydrostatic equilibrium by using the acceleration of gravity and the
    temperature field (see Eq. 9 in https://doi.org/10.1121/10.0002912),
    and implements it as a subclass of *MathFunc*.

    **Class attributes:**

    Attributes
    ----------
    z0: float
        Reference height (in :math:`m`).
    g: MathFunc
        The acceleration due to gravity (in :math:`ms^{-2}`), implemented as a MathFunc.
    rho0: float
        Value for the air density (in :math:`kgm^{-3}`) at the reference height (z0).
    t: MathFunc
        The temperature field (in :math:`K`), implemented as a *MathFunc*.
    air_properties: AirProperties
        The properties of air, implemented as a *AirProperties* object.
    """

    def __init__(self, z0: float, g: MathFunc, rho0: float, t: MathFunc, air_properties: AirProperties):
        """

        Parameters
        ----------
        z0: float
            Reference height (in :math:`m`).
        g: MathFunc
            The acceleration due to gravity (in :math:`ms^{-2}`) implemented as a MathFunc.
        rho0: float
            Value for the air density (in :math:`kgm^{-3}`) at the reference height (z0).
        t: MathFunc
            The temperature field (in :math:`K`), implemented as a *MathFunc*.
        air_properties: AirProperties
            The properties of air implemented as a *AirProperties* object.

        """
        self.z0 = z0                # Reference Height [m]
        self.rho0 = rho0            # Air Density at Reference Height [Kg/m^3]
        self.g = g
        self.t = t                  # Atmospheric Temperature Object (Type: MathFunc) [K]
        self.air_properties = air_properties    # Atmospheric Properties Object (Type: AirProperties)

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Computes the value of air density with hydrostatic relations at z(s).

        Parameters
        ----------
        z: Arraylike
            The input height(s) (in :math:`m`).

        Returns
        -------
        float
            The value of air density at z(s).
        """

        t0 = self.t.value(self.z0)                  # Air Temperature at Reference Height [K]
        invt = PowerFunc(self.t, ConstFunc(-1))     # Inverse of Atmospheric Temperature (MathFunc) [K^-1]

        # Integrating from Reference Height to Desired Height (z)
        z = np.asarray(z)
        if z.ndim == 0:
            value = invt.integrate([self.z0, z])
        else:
            value = [invt.integrate([self.z0, coord]) for coord in z]

        r = self.air_properties.r.value(z)                  # Local Specific Heats Ratios of Air (Type: Float) [-]
        g = self.g.value(z)                                 # Local Acceleration of Gravity (Type: Float) [ms^-2]
        t = self.t.value(z)                                 # Local Atmospheric Temperature (Type: Float) [K]
        value = (self.rho0 * t0/t) * np.exp(-(g/r)*value)   # Local Atmospheric Density (eq. 9 in Ref, [2])
        return value

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Computes the derivative with respect to height of air density using hydrostatic relations.

        Parameters
        ----------
        z:ArrayLike
            The input height(s) (in :math:`m`).

        Returns
        -------
        np.ndarray
            The derivative with respect to height of the air density at z(s).
        """
        
        r = self.air_properties.r.value(z)  # Local Specific Heats Ratios of Air (Type: Float) [-]
        g = self.g.value(z)  # Local Acceleration of Gravity (Type: Float) [ms^-2]

        val = -(self.value(z)/self.t.value(z)) * ((g/r) + self.t.dvalue(z))
        return val

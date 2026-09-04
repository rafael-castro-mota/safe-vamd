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
    A class that computes the density in hydrostatic equilibrium with the acceleration of gravity and the
    temperature profile and implements it as a subclass of MathFunc:

    Attributes
    ----------
    z0: float
        Reference height (in meters).
    g: MathFunc
        The acceleration of gravity (in m/s) implemented as a MathFunc.
    rho0: float
        Value for the air density (in kg/m^3) at the reference height (z0).
    t: MathFunc
        The air temperature (in Kelvins) field implemented as a MathFunc.
    air_properties: AirProperties
        The air properties (specific ratio of heats and perfect gas constance) implemented as a AirProperties object.
    """

    def __init__(self, z0: float, g: MathFunc, rho0: float, t: MathFunc, air_properties: AirProperties):
        """
        Initializes an instance of RhoStratAtmo

        """
        self.z0 = z0                # Reference Height [m]
        self.rho0 = rho0            # Air Density at Reference Height [Kg/m^3]
        self.g = g
        self.t = t                  # Atmospheric Temperature Object (Type: MathFunc) [K]
        self.air_properties = air_properties    # Atmospheric Properties Object (Type: AirProperties)

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Computes the value of air density with hydrostatic relations.

        Parameters
        ----------
        z: Arraylike
            The height(s) (in meters) at which the value of the density should be computed.

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
            #value = invt.glq_quadrature([self.z0, z])
            value = invt.integrate([self.z0, z])
            #value = quad(invt.value, self.z0, float(z))[0]
        else:
            #value = [invt.glq_quadrature([self.z0, coord]) for coord in z]
            value = [invt.integrate([self.z0, coord]) for coord in z]
            #value = [quad(invt.value, self.z0, float(coord))[0] for coord in z]

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
            The height(s) (in meters) at which the derivative should be computed.

        Returns
        -------
        np.ndarray
            The derivative with respect to height of air density at z(s).
        """
        
        r = self.air_properties.r.value(z)  # Local Specific Heats Ratios of Air (Type: Float) [-]
        g = self.g.value(z)  # Local Acceleration of Gravity (Type: Float) [ms^-2]

        val = -(self.value(z)/self.t.value(z)) * ((g/r) + self.t.dvalue(z))
        return val

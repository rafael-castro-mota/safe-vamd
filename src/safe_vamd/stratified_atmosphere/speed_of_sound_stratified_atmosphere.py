import numpy as np

from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class CStratAtmo(MathFunc):
    """
    A class that computes the speed of sound and implements it as a subclass of *MathFunc*.

    **Class attributes:**

    Attributes
    ----------
    t: MathFunc
        The temperature field (in :math:`K`), implemented as a *MathFunc*.
    air_properties: AirProperties
        The properties of air, implemented as a *AirProperties* object.
    """

    def __init__(self, t, air_properties):
        """

        Parameters
        ----------
        t: MathFunc
            The temperature field (in :math:`K`), implemented as a *MathFunc*.
        air_properties: AirProperties
            The properties of air, implemented as a *AirProperties* object.
        """

        self.t = t                  # Atmospheric Temperature Object (Type: MathFunc) [K]
        self.air_properties = air_properties    # Atmospheric Properties Object (Type: AirProperties) [-]

    def value(self, z: float) -> float:
        """
        Computes the value of the speed of sound at the input height.

        Parameters
        ----------
        z: float
            The input height (in :math:`m`).

        Returns
        -------
        float
            The value of the speed of sound at the input height.
        """

        # Computing real part of the speed of sound
        t = self.t.value(z)                         # Local Atmospheric Temperature  [K]
        r = self.air_properties.r.value(z)                # Local Perfect Gas Constant of Air [K]
        gamma = self.air_properties.gamma.value(z)        # Local Ratio of Specific Heats  [-]
        val = np.sqrt(gamma * r * t)  # Local Real Part of Speed of Sound (Type: Float) [ms^-1]

        return val

    def dvalue(self, z: float) -> float:
        """
        Computes the derivative with respect to height of the speed of sound at the input height.

        Parameters
        ----------
        z: float
           The input height(s) (in :math:`m`).

        Returns
        -------
        float
            The value of the derivative with respect to height of the speed of sound at the input height.
        """

        dt = self.t.dvalue(z)           # Vertical Derivative of Local Atmospheric Temperature [K]
        r = self.air_properties.r.value(z)      # Local Atmospheric Perfect Gas Constant of Air [K]
        gamma = self.air_properties.gamma.value(z)  # Local Atmospheric Specific Heats Ratio of Air [-]

        # Derivative of Speed of Sound (real part, eq. 36 b in Ref. [1])
        val = dt * (gamma * r) / (2 * np.real(self.value(z)))
        return val

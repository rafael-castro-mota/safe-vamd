from safe_vamd.background_atmospheric_field.air_properties import AirProperties
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class PStratAtmo(MathFunc):
    """
    A class that computes the air pressure in hydrostatic equilibrium with the air density and
    temperature fields, and implements it as a subclass of *MathFunc*.

    **Class attributes:**

    Attributes
    ----------
    rho: MathFunc
        The density field (in :math:`kgm^{-3}`), implemented as a MathFunc.
    t: MathFunc
        The temperature field (in :math:`K`), implemented as a MathFunc.
    air_properties: AirProperties
        The properties of air, implemented as a *AirProperties* object.
    """
    def __init__(self, rho: MathFunc, t: MathFunc, air_properties: AirProperties) -> None:
        """

        Parameters
        ----------
        rho: MathFunc
            The density field (in :math:`kgm^{-3}`), implemented as a MathFunc.
        t: MathFunc
            TThe temperature field (in :math:`K`), implemented as a MathFunc.
        air_properties: AirProperties
            The properties of air, implemented as a *AirProperties* object.
        """

        self.rho = rho              # Atmospheric Density Object (Type: MathFunc) [Kgm^-3]
        self.t = t                  # Atmospheric Temperature Object (Type: MathFunc) [K]
        self.air_properties = air_properties    # Atmospheric Properties Object (Type: AirProperties) [J/(Kg*K)]

    def value(self, z: float) -> float:
        """
        Computes the value of air pressure at the input height, using the ideal gas law.

        Parameters
        ----------
        z: float
            The input height (in :math:`m`).

        Returns
        -------
        float
            The value of air pressure at the input height.
        """
        r = self.air_properties.r.value(z)
        value = r * self.rho.value(z) * self.t.value(z)
        return value

    def dvalue(self, z: float) -> float:
        """
        Computes the derivative with respect to height of the air pressure at the input height, using the ideal gas law.

        Parameters
        ----------
        z: float
            The input height (in :math:`m`).

        Returns
        -------
        float
            The value of the derivative with respect to height at the input height.
        """

        r = self.air_properties.r.value(z)
        t = self.t.value(z)
        dt = self.t.dvalue(z)
        rho = self.rho.value(z)
        drho = self.rho.dvalue(z)
        value = r * (t * drho + rho * dt)

        return value

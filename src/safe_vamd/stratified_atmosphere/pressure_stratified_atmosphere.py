from safe_vamd.background_atmospheric_field.air_properties import AirProperties
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class PStratAtmo(MathFunc):
    """
    A class that computes the air pressure in hydrostatic equilibrium with the air density pressure
    temperature profiles and implements it as a subclass of MathFunc:

    Attributes
    ----------
    rho: MathFunc
        The air density field implemented as a MathFunc object.
    t: MathFunc
        The temperature field implemented as a MathFunc object.
    air_properties: AirProperties
        The air properties (specific ratio of heats and perfect gas constance) implemented as a AirProperties object.
    """
    def __init__(self, rho: MathFunc, t: MathFunc, air_properties: AirProperties) -> None:
        """
        Initializes an instance of the PStratAtmo class.
        """

        self.rho = rho              # Atmospheric Density Object (Type: MathFunc) [Kgm^-3]
        self.t = t                  # Atmospheric Temperature Object (Type: MathFunc) [K]
        self.air_properties = air_properties    # Atmospheric Properties Object (Type: AirProperties) [J/(Kg*K)]

    def value(self, z: float) -> float:
        """
        Computes the value of air pressure with the perfect gas law.

        Parameters
        ----------
        z: float
            The height at which the value of the air pressure should be computed.

        Returns
        -------
        float
            The value of air pressure at the desired height (z).
        """
        r = self.air_properties.r.value(z)
        value = r * self.rho.value(z) * self.t.value(z)
        return value

    def dvalue(self, z: float) -> float:
        """
        Computes the derivative with respect to height of the air pressure using the perfect gas law.

        Parameters
        ----------
        z: float
            The height at which the value of the derivative with respect to height should be computed.

        Returns
        -------
        float
            The value of the derivative with respect to height at the desired height (z).
        """

        r = self.air_properties.r.value(z)
        t = self.t.value(z)
        dt = self.t.dvalue(z)
        rho = self.rho.value(z)
        drho = self.rho.dvalue(z)
        value = r * (t * drho + rho * dt)

        return value

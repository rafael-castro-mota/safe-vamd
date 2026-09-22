from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from safe_vamd.background_atmospheric_field.air_properties import AirProperties


class BackgroundField:
    r"""
    A class that holds the fields for the atmospheric background quantities

    **Class attributes:**

    Attributes
    ----------
    air_properties: AirProperties
        The properties of air, implemented as a *AirProperties* object.
    g: MathFunc
        The acceleration due to gravity (in :math:`ms^{-2}`), implemented as a MathFunc.
    rho: MathFunc
        The density field (in :math:`kgm^{-3}`), implemented as a MathFunc.
    rh: MathFunc
        The relative humidity field (in :math:`\%`), implemented as a MathFunc.
    p: MathFunc
        The absolute pressure field (in :math:`Pa`), implemented as a MathFunc.
    t: MathFunc
        The temperature field (in :math:`K`), implemented as a MathFunc.
    vx: MathFunc
        The wind velocity field (in :math:`ms^{-1}`) in the range direction (:math:`x`), implemented as a MathFunc.
    vy: MathFunc
        The wind velocity field (in :math:`ms^{-1}`) in the cross-plane direction (:math:`y`),
        implemented as a MathFunc.
    vz: MathFunc
        The wind velocity field (in :math:`ms^{-1}`) in the vertical direction (:math:`z`), implemented as a MathFunc.
    c: MathFunc
        The speed of sound field (in :math:`ms^{-1}`), implemented as a MathFunc.
    """
    def __init__(self, air_properties: AirProperties, g: MathFunc, rho: MathFunc, rh: MathFunc, p: MathFunc, t: MathFunc
                 , vx: MathFunc, vy: MathFunc, vz: MathFunc, c: MathFunc):

        """

        Parameters
        ----------
        air_properties: AirProperties
            The properties of air implemented as a *AirProperties* object.
        g: MathFunc
            The acceleration due to gravity (in :math:`ms^{-2}`), implemented as a MathFunc.
        rho: MathFunc
            The density field (in :math:`kgm^{-3}`), implemented as a MathFunc.
        rh: MathFunc
            The relative humidity field (in :math:`%`), implemented as a MathFunc.
        p: MathFunc
            The absolute pressure field (in :math:`Pa`), implemented as a MathFunc.
        t: MathFunc
            The temperature field (in :math:`K`), implemented as a MathFunc.
        vx: MathFunc
            The wind velocity field (in :math:`ms^{-1}`) in the range direction (:math:`x`), implemented as a MathFunc.
        vy: MathFunc
            The wind velocity field (in :math:`ms^{-1}`) in the cross-plane direction (:math:`y`), implemented as
            a MathFunc.
        vz: MathFunc
            The wind velocity field (in :math:`ms^{-1}`) in the vertical direction (:math:`z`), implemented as
            a MathFunc.
        c: MathFunc
            The speed of sound field (in :math:`ms^{-1}`), implemented as a MathFunc.
        """

        self.air_properties = air_properties
        self.g = g
        self.rho = rho
        self.rh = rh
        self.p = p
        self.t = t
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.c = c


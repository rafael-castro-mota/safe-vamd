

class BackgroundField:
    """
    A class that holds the fields for the atmospheric background quantities

    Attributes
    ----------
    air_properties: AirProperties
        The air properties (specific ratio of heats and perfect gas constance) implemented as a AirProperties object.
    g: MathFunc
        The acceleration of gravity (in m/s^2) implemented as a MathFunc.
    rho: MathFunc
        The air density field (in kg/m°3) implemented as a MathFunc.
    rh: MathFunc
        The relative humidity field (in %) implemented as a MathFunc.
    p: MathFunc
        The pressure field (in Pa) implemented as a MathFunc.
    t: MathFunc
        The temperature field (in K) implemented as a MathFunc.
    vx: MathFunc
        The wind velocity (in m/s) in the range direction (x) implemented as a MathFunc.
    vy: MathFunc
        The wind velocity (in m/s) in the cross-plane direction (y) implemented as a MathFunc.
    vz: MathFunc
        The wind velocity (in m/s) in the vertical direction (z) implemented as a MathFunc.
    c: MathFunc
        The speed of sound field (in m/s) implemented as a MathFunc.
    """
    def __init__(self, air_properties, g, rho, rh, p, t, vx, vy, vz, c):

        self.air_properties = air_properties
        self.g = g
        self.rho = rho  # Density Field  [Kg/m^3]
        self.rh = rh    # Relative Humidity Field [%]
        self.p = p      # Pressure Field [Pa]
        self.t = t      # Temperature Field  [k]
        self.vx = vx    # X-Velocity Field  [m/s]
        self.vy = vy    # Y-Velocity Field  [m/s]
        self.vz = vz    # Z-Velocity Field  [m/s]
        self.c = c      # Speed of Sound Field [m/s]

        
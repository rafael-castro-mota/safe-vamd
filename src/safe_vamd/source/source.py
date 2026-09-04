

class Source:
    """
    A class to hold information about a monopole.

    Attributes
    ----------
    w: MathFunc
        The excitation frequency (in radeans per second).
    height: float
        The height (in meters) at which the monopole is placed.
    amplitude: float
        The amplitude (in kg/(m^-3 s)) of the monopole.
    """

    def __init__(self, w: float, height: float, amplitude: float) -> None:
        """
        Initializes an instance of class Source.

        """

        self.w = w                          # Frequency of the Point Source [rads/s]
        self.height = height                # Height of the Point Source [m]
        self.amplitude = amplitude          # Amplitude of the Point Source [kg/(m^-3 s)]



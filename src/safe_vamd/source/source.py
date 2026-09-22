

class Source:
    """
    A class to hold information about a monopole.

    **Class attributes:**

    Attributes
    ----------
    w : MathFunc
        The excitation frequency (in :math:`rads^{-1}`).
    height : float
        The height (in :math:`m`) at which the monopole is placed.
    amplitude : float
        The amplitude (in :math:`kgm^{-3}s^{-1}`) of the monopole.
    """

    def __init__(self, w: float, height: float, amplitude: float) -> None:
        """

        Parameters
        ----------
        w : MathFunc
            The excitation frequency (in :math:`rads^{-1}`).
        height : float
            The height (in :math:`m`) at which the monopole is placed.
        amplitude : float
            The amplitude (in :math:`kgm^{-3}s^{-1}`) of the monopole.
        """

        self.w = w                          # Frequency of the Point Source [rads/s]
        self.height = height                # Height of the Point Source [m]
        self.amplitude = amplitude          # Amplitude of the Point Source [kg/(m^-3 s)]



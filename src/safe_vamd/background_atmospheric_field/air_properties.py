from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class AirProperties:
    """
    A class to hold air properties (ratio of specific heats and perfect gas constant)

    Attributes
    ----------
    gamma: MathFunc
        The ratio of specific heats implemented as a MathFunc object.
    r: float
        The perfect gas constant implemented as a MathFunc object.

    """
    def __init__(self, gamma: MathFunc, r: MathFunc) -> None:
        """
        Initializes an instance of class AirProperties.
        """

        self.gamma = gamma  # Specific Heats Ratio
        self.r = r          # Perfect Gas Constant

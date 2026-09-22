from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class AirProperties:
    r"""
    A class to hold the ratio of specific heats :math:`\gamma` and ideal gas constant :math:`R` of air.

    **Class attributes:**

    Attributes
    ----------
    gamma: MathFunc
        The ratio of specific heats :math:`\gamma` (unitless) implemented as a *MathFunc* object.
    r: MathFunc
        The ideal gas constant :math:`R` (in :math:`Jkg^{-1}K^{-1}`) implemented as a *MathFunc* object.

    """
    def __init__(self, gamma: MathFunc, r: MathFunc) -> None:
        r"""
        Parameters
        ----------
        gamma: MathFunc
            The ratio of specific heats :math:`\gamma` (unitless) implemented as a *MathFunc* object.
        r: MathFunc
            The ideal gas constant :math:`R` (in :math:`Jkg^{-1}K^{-1}`) implemented as a *MathFunc* object.

        """

        self.gamma = gamma
        self.r = r

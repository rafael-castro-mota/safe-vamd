import numpy as np
from numpy.typing import ArrayLike

from safe_vamd.source.source import Source
from safe_vamd.background_atmospheric_field.background_field import BackgroundField
from safe_vamd.mathematical_functions.mathematical_func import MathFunc


class WaveNumberWithAbsorption(MathFunc):
    """
    The wave number with an added imaginary term to account for air absorption, implemented as subclass of MathFunc:
    The air absorption is computed with the methodology in https://doi.org/10.1121/1.412989.

    Attributes
    ----------
     source: Source
        An instance of the Source class with monopole source information.
    backfield: : BackgroundField
        An instance of the BackgroundField class representing the atmospheric background field.
    c: MathFunc
        Speed of sound (in m/S).
    """

    def __init__(self, source: Source, backfield: BackgroundField, c: MathFunc):
        """
        Initializes an instance of the WaveNumberWithAbsorption.
        """

        self.source = source
        self.backfield = backfield
        self.c = c

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the wave number with the added air absorption at z(s)

        Parameters
        ----------
        z: ArrayLike
         Height(s) (in meters).
        Returns
        -------
        np.ndarray
         Value(s) of the wave number with the added air absorption at z(s)
        """

        z = np.asarray(z)
        # Computing real part of the speed of sound
        w = self.source.w                       # Frequency of the Point Source (Type: Float) [rads/s]
        c = self.c.value(z)                     # Local Speed of Sound (tYPE: Float) [m/s^2]
        t = self.backfield.t.value(z)           # Local Atmospheric Temperature (Type: Float) [K]
        p = self.backfield.p.value(z)
        hr = self.backfield.rh.value(z)         # Local Relative Humidity (Type: Float) [%]

        val = w / c     # Real part of Local Wave Number (Type: Float) [-]

        # Calculating Imaginary Part
        fs = self.source.w / (2 * np.pi)     # Frequency of the Point Source (Type: Float) [Hz]
        t0 = 293.15  # Reference Atmospheric Temperature (Type: Float) [K]
        t01 = 273.16  # Triple-Point Isotherm Temperature (Type: Float) [k]
        ps = p / 101325  # Local Atmospheric Pressure (Type: Float) [atm]
        ps0 = 1  # Reference Atmospheric Pressure [atm]

        psat = ps0 * 10 ** (-6.8346 * (t01 / t) ** 1.261 + 4.6151)  # Local Saturation Pressure (Type: Float) [atm]
        h = ps0 * (hr / ps) * (psat / ps0)  # Local Absolute Humidity (Type: Float) [%]

        f = fs / ps  # F factor
        fro = (1 / ps0) * (24 + 4.04 * (10 ** 4) * h * ((0.02 + h) / (0.391 + h)))  # FrO factor [Hz/atm]
        # Computing FrN factor [Hz/atm]
        frn = (1 / ps0) * np.sqrt(t0 / t) * (9 + 280 * h * np.exp(-4.17 * ((t0 / t) ** (1 / 3) - 1)))

        # Computing alpha factor [Neper/m]
        alfa = ps * (f ** 2 / ps0) * ((1.84 * 10 ** (-11)) * np.sqrt(t / t0) +
                                      ((t / t0) ** (-5 / 2)) * (
                                              0.01278 * ((np.exp(-2239.1 / t)) / (fro + (f ** 2) / fro))) +
                                      0.1068 * (np.exp(-3352 / t) / (frn + (f ** 2) / frn)))

        # Complex wave number
        val = val - alfa * 1j

        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative with respect to height of the wave number at z(s),
        disregarding the air absorption imaginary term

        Parameters
        ----------
        z: ArrayLike
            Height(s) (in meters).
        Returns
        -------
        np.ndarray
            Value(s) of the derivative with respect to height of the wave number at z(s), disregarding the
             air absorption imaginary term
        """

        c = self.backfield.c.value(z)
        dc = self.backfield.c.dvalue(z)
        w = self.source.w
        dw = 0
        val = dw * np.real(c) - w * np.real(dc)
        val = val / (np.real(c) ** 2)
        return val

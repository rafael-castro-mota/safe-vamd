
import numpy as np
import scipy
from numbers import Complex
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class MathFunc(ABC):
    """
    Abstract base class for a single-variable mathematical function.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes an instance of MathFunc.
        """
        pass

    @abstractmethod
    def value(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the function at x(s).
        """
        pass

    @abstractmethod
    def dvalue(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the function at x(s).
        """
        pass

    def integrate(self, inter: ArrayLike) -> complex:
        """
        Integrates the mathematical function from inter[0] to inter[1].
        Parameters
        ----------
        inter: ArrayLike
            Integration interval (real numbers only).
        Returns
        -------
        complex
            The complex-valued integration for the input interval.
        """

        integration_value_real = scipy.integrate.quad(lambda x: np.real(self.value(x)), inter[0], inter[1])[0]
        integration_value_imag = scipy.integrate.quad(lambda x: np.imag(self.value(x)), inter[0], inter[1])[0]

        return integration_value_real + 1j * integration_value_imag

    def glq_quadrature(self, inter: ArrayLike) -> float | complex:
        """
        Computes the integral of the function in a desired interval using a 3-point Gauss–Legendre quadrature.

        Parameters
        ----------
        inter : array_like of shape (2,1)
           Integration interval [a, b], where a < b.

        Returns
        -------
        float | complex
           Value of the integral over [a, b].
        """

        # Gauss legendre Points and Weights (for [-1,1] interval)
        glq_weights = np.array([0.555555555555555, 0.888888888888888, 0.555555555555555])
        glq_points = np.array([-0.77459666924148, 0, 0.77459666924148])

        # Converting Gauss Legendre points [-1,1] into the interval
        inter = np.asarray(inter)
        glq_points = (abs(inter[1] - inter[0]) / 2) * glq_points + (inter[1] + inter[0]) / 2

        # Executing the Gauss Legendre Quadrature
        value = np.sum(glq_weights * self.value(glq_points))

        # Using the integral property of linearly mapped coordinates
        value = value * abs(inter[1] - inter[0]) / 2
        return value

    # Plotting function over the desired interval
    def plot(self, inter: ArrayLike, n_inter: int = 1000, ylab: str = "Height [m]", xlab: str = "Value [-]",
             mk: bool = 0, real_color: str = 'black', imag_color: str = 'blue') -> None:

        """
        plots the real and imaginary part of the MathFunc object over a height interval.

        Parameters
        ----------
        inter : ArrayLike of shape (2,1)
            Plotting height interval [a, b], where a < b.
        n_inter : int
            Number of equally distanced points to plot.
        ylab : str
            Y-axis label.
        xlab : str
            X-axis label.
        mk: bool
            Whether the plot is shown (mk=1) or not (mk=0).
        real_color
            Color of the plotted line for the real part.
        imag_color
            Color of the plotted line for the imaginary part.

        Returns
        -------
        None
        """

        x = np.linspace(inter[0], inter[1], n_inter)
        plt.plot(np.real(self.value(x)), x, color=real_color)
        plt.plot(np.imag(self.value(x)), x, color=imag_color)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.legend(['real', 'imaginary'])

        if mk == 1:
            plt.show()


import numpy as np
from typing import Sequence
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from safe_vamd.solver.acoustic_field import AcousticField


class ModeFilter:
    """
    A class that does mode filtering operations on an AcousticField object.

    Parameters
    ----------
    acoustic_field : AcousticField
        An AcousticField object whose modes are going to be filtered
    """

    def __init__(self, acoustic_field: AcousticField) -> None:
        """
        Initializes an instance of ModeFilter.
        """

        self.acoustic_field = acoustic_field                # Acoustic Pressure Field (AcousticField object)

    def sort_by_imag_part(self, side: str = "downwind") -> None:
        """
        Sorts the modes from least attenuated to most attenuated. In the case of downwind modes, the least attenuated
        mode is the one with the lowest valued imaginary part. For the upwind modes, it is the one with the highest.

        Parameters
        ----------
        side: str, optional
          Set of modes used to sort. (side='downwind' sorts the downwind modes, side='upwind'  the upwind modes)

        Returns
        -------
        None
        """

        if side == "downwind":
            modes = np.array(self.acoustic_field.downwind_modes)
            eigvals = [mode.eigval for mode in modes]
            indexes = np.flip(np.argsort(np.imag(eigvals)).ravel())
            self.acoustic_field.downwind_modes = modes[indexes]
        else:
            modes = np.array(self.acoustic_field.upwind_modes)
            eigvals = [mode.eigval for mode in modes]
            indexes = np.argsort(np.imag(eigvals)).ravel()
            self.acoustic_field.upwind_modes = modes[indexes]

    def filter_by_energy(self, inter, energy_concentration, side="downwind"):
        if side == "downwind":
            modes = self.acoustic_field.downwind_modes
        else:
            modes = self.acoustic_field.upwind_modes

        # Select nodes that belong to the interval
        coords = modes[-1].z_coords
        inner_indexes = np.argwhere(np.logical_and(inter[0] <= coords, coords <= inter[1])).ravel()

        # Calculating how strong is the mode in the desired area
        selected_modes = []
        unselected_modes = []
        for mode in modes:
            eigvec = np.array(mode.eigvec)
            duct_metric = linalg.norm(eigvec[inner_indexes]) ** 2 # Energy of the mode shape in the considered interval
            total_metric = linalg.norm(eigvec) ** 2 # Total energy of the mode shape
            if duct_metric/total_metric >= energy_concentration:
                selected_modes.append(mode)
            else:
                unselected_modes.append(mode)

        if side == "downwind":
            self.acoustic_field.downwind_modes = selected_modes
        else:
            self.acoustic_field.upwind_modes = selected_modes

        return unselected_modes

    def filter_by_real_part(self, inter: Sequence[float], side: str = "downwind") -> None:
        """
        Filters out the modes whose eigenvalues' real part are not within a desired interval.

        Parameters
        ----------
        inter : Sequence[float]
            The admissible interval for the eigenvalue's real part.
        side: str, optional
         Set of modes to filter from. (side='downwind' filters from the downwind modes,
          side='upwind' from the upwind modes)

        Returns
        -------
        None
        """

        if side == "downwind":
            modes = np.array(self.acoustic_field.downwind_modes)
        else:
            modes = np.array(self.acoustic_field.upwind_modes)

        eigvals = [mode.eigval for mode in modes]
        cond = np.logical_and(np.real(eigvals) >= inter[0], np.real(eigvals) <= inter[1])
        indexes = np.argwhere(cond).ravel()

        if side == "downwind":
            self.acoustic_field.downwind_modes = modes[indexes]
        else:
            self.acoustic_field.upwind_modes = modes[indexes]

    def filter_by_imag_part(self, inter: Sequence[float], side: str = "downwind") -> None:
        """
        Filters out the modes whose eigenvalues' imaginary part are not within a desired interval.

        Parameters
        ----------
        inter : Sequence[float]
            The admissible interval for the eigenvalue's imaginary part.
        side: str, optional
            Set of modes to filter from. (side='downwind' filters from the downwind modes,
            side='upwind' from the upwind modes)

        Returns
        -------
        None
        """

        if side == "downwind":
            modes = np.array(self.acoustic_field.downwind_modes)
        else:
            modes = np.array(self.acoustic_field.upwind_modes)

        eigvals = [mode.eigval for mode in modes]
        cond = np.logical_and(np.imag(eigvals) >= inter[0], np.imag(eigvals) <= inter[1])
        indexes = np.argwhere(cond).ravel()

        if side == "downwind":
            self.acoustic_field.downwind_modes = modes[indexes]
        else:
            self.acoustic_field.upwind_modes = modes[indexes]

    def plot_eigenvalues(self, upwind_color: str = 'black', upwind_label: str = 'upwind modes',
                         downwind_color: str = 'red', downwind_label: str = 'downwind modes', mk: bool = False) -> None:
        """
        Plots the eigenvalues for the upwind and downwind modes in the real-complex plane.

        Parameters
        ----------
        upwind_color: str, optional
            Color for the upwind eigenvalue markers.
        upwind_label: str, optional
            Label for the plotted upwind eigenvalues.
        downwind_color: str, optional
            Color for the downwind eigenvalue markers.
        downwind_label: str, optional
            Label for the plotted downwind eigenvalues.
        mk: bool, optional
            Whether to show the plot (mk=True) or not (mk=False)

        Returns
        -------
        None
        """

        upwind_eigvals = np.array([mode.eigval for mode in self.acoustic_field.upwind_modes]).ravel()
        downwind_eigvals = np.array([mode.eigval for mode in self.acoustic_field.downwind_modes]).ravel()

        plt.scatter(np.real(upwind_eigvals), np.imag(upwind_eigvals), color=upwind_color, label=upwind_label)
        plt.scatter(np.real(downwind_eigvals), np.imag(downwind_eigvals), color=downwind_color, label=downwind_label)

        if mk:
            plt.title('Eigenvalues', fontsize=12)
            plt.xlabel('Real(λ)', fontsize=12)
            plt.ylabel('Imag(λ)', fontsize=12)
            plt.legend(fontsize=10)
            plt.show()


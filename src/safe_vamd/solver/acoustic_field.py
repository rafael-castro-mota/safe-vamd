import numpy as np
from typing import Sequence

from safe_vamd.solver.mode import Mode


class AcousticField:
    """
    A class that represents a two-dimensional acoustic field comprised of a superposition of downwind or upwind
    vertical atmospheric (VA) modes. At the 'right' of the range origin (x>0), the field is assumed to be comprised
    of only downwind modes and on the 'left' (x<0) of only upwind modes.

    Attributes
    ----------
    downwind_modes: Sequence[Mode]
        The downwind VA modes.
    upwind_modes: Sequence[Mode]
        The upwind VA modes.

    """
    def __init__(self, downwind_modes: Sequence[Mode] = None, upwind_modes: Sequence[Mode] = None):
        """
        Initializes an instance of AcousticField.
        """

        self.downwind_modes = downwind_modes    # Acoustic Modes (Type: Array of Mode objects)
        self.upwind_modes = upwind_modes        # Left-Moving Acoustic Modes

    def value_2d(self, x: float, z: float) -> complex:
        """
        Evaluates the acoustic field at (x, z)

        Parameters
        ----------
        x: float
            Range (in meters).
        z: float
            Height (in meters).

        Returns
        -------
        The complex-valued amplitude of the acoustic field at (x, z)
        """

        if x >= 0:
            modes = self.downwind_modes
        else:
            modes = self.upwind_modes

        mode_contributions = np.array([mode.amplitude * mode.value_2d(x, z) for mode in modes])
        value = np.sum(mode_contributions, axis=0)
        return value

    def value_2d_with_3d_spread(self, x: float, z: float, x_ref: float):
        """
        Evaluates the acoustic field at (x, z) with spherical spreading accounted for
         with a correction factor (1/√(x/x_ref)).

        Parameters
        ----------
        x: float
            Range (in meters).
        z: float
            Height (in meters).
        x_ref: float
            Reference range (in meters).

        Returns
        -------
        The complex-valued amplitude of the acoustic field at (x, z) with spherical spreading accounted for.
        """

        if x >= 0:
            modes = self.downwind_modes
        else:
            modes = self.upwind_modes

        mode_contributions = np.array([(1 / np.sqrt(x / x_ref)) * mode.amplitude * mode.value_2d(x, z) for mode in modes])
        value = np.sum(mode_contributions, axis=0)
        return value

    def save_modes_in_txt(self, side: str = "downwind") -> None:
        """
        Saves either the downwind or upwind modes into a txt file (side='downwind' saves the downwind modes
        and side='upwind' saves the upwind modes). By default, it saves the downwind modes.

        Parameters
        ----------
        side: str, optional
          Which set of modes to save (side='downwind' saves the downwind modes and side='upwind' saves the upwind modes)


        Returns
        -------
        The complex-valued amplitude of the acoustic field at (x, z) with spherical spreading accounted for.
        """
        if side == "downwind":
            modes = self.downwind_modes
            file_name = "downwind_modes.txt"
        else:
            modes = self.upwind_modes
            file_name = "upwind_modes.txt"

        # Saving the modes
        z_coords = np.array(modes[0].z_coords)
        n_nodes = len(z_coords)

        # Saving downwind modes
        n_modes = len(modes)
        data = np.zeros((n_nodes+3, n_modes+1), dtype=complex)

        i = 0
        data[:-3, 0] = z_coords
        for mode in modes:
            data[-3, i + 1] = mode.ref_wn
            data[-2, i + 1] = mode.eigval
            data[-1, i + 1] = mode.amplitude
            data[:-3, i + 1] = mode.eigvec
            i = i + 1

        np.savetxt(file_name, data)

    def load_modes_from_txt(self, mode_path: str, side: str = "downwind") -> None:
        """
        Loads the VA modes from a text file. Setting side='downwind' loads the downwind modes from the files and
        side='upwind' loads the upwind modes.

        Parameters
        ----------
        mode_path: str
            Path to the text file with modes.
        side: str, optional
         Selects the set of modes to be loaded (side='downwind' loads the downwind modes
         and side='upwind' the upwind ones)

        Returns
        -------
        None
        """

        data = np.loadtxt(mode_path, dtype=complex)
        n_modes = len(data[0, :])-1
        heights = np.real(data[0:-3, 0]).ravel()

        modes = []
        for i in range(1, n_modes):
            eigvec = data[0:-3, i].ravel()
            ref_wn = data[-3, i]
            eigval = data[-2, i]
            amplitude = data[-1, i]

            modes.append(Mode(amplitude, heights, eigval, eigvec, ref_wn))

        if side == "downwind":
            self.downwind_modes = modes
        else:
            self.upwind_modes = modes

    def get_plot_data(self, rangos: Sequence[float], side='downwind') -> tuple:

        """
        Generates plot data by defining a grid with the prescribed ranges and the heights of the
         nodes in the computational mesh and evaluates the acoustic field at those points.

        Parameters
        ----------
        rangos: Sequence[float]
            Ranges (in meters) at which to evaluate the acoustic field.
        side: str, optional
          Set of modes used to compute the plot data (side='downwind' for downwind modes and
           side='upwind' for the upwind modes)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing matrices for the ranges,
             height and complex-valued acoustic pressure amplitudes at each grid point.
        """

        if side == 'downwind':
            modes = self.downwind_modes
        else:
            modes = self.upwind_modes

        z_coords = np.array(modes[0].z_coords).ravel()
        x = rangos * np.ones((len(z_coords), len(rangos)))
        z = np.transpose(z_coords * np.ones((len(rangos), len(z_coords))))
        p = np.zeros((len(z_coords), len(rangos)))
        for mode in modes:
            eigvec = mode.amplitude * mode.eigvec
            range_shift = np.exp(-1j*mode.eigval*mode.ref_wn*np.array(rangos)) * np.ones((len(z_coords), len(rangos)))
            p = p + eigvec.reshape((len(eigvec), 1)) * range_shift
        return x, z, p

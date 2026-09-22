import numpy as np
from typing import Sequence

from safe_vamd.solver.mode import Mode


class AcousticField:
    r"""
    A class that represents a two-dimensional acoustic field comprised of a superposition of downwind or upwind
    vertical atmospheric (VA) modes. At the 'right' of the range origin (x>0), the field is assumed to be comprised
    of only downwind modes and on the 'left' (x<0) of only upwind modes.

    **Class attributes:**

    Attributes
    ----------
    downwind_modes: Sequence[Mode]
        The downwind Vertical Atmospheric (VA) modes.
    upwind_modes: Sequence[Mode]
        The upwind Vertical Atmospheric (VA) modes.

    Notes
    -----
    The two-dimensional acoustic field :math:`p^{'}` can be represented by:

    .. math::
        p^{'}(f, x, z) =
        \begin{cases}
            \sum_{m=1}^{N^{+}} A_m^{+}p^{+}_m(z)e^{-ik^{+}_{m}x}, & x \geq 0 \\
            \sum_{n=1}^{N^{-}} A_n^{-}p^{-}_n(z)e^{-ik^{-}_{n}x}, & x < 0
        \end{cases}
    ,where :math:`A^{\pm}`, :math:`p^{\pm}` and :math:`k^{\pm}` are the complex-valued amplitude, mode shape,
    and wave number of a mode, with downwind modes of the order :math:`m` denoted by the superscript :math:`+` and
    upwind modes of order :math:`n` denoted by the superscript :math:`-`.
    """
    def __init__(self, downwind_modes: Sequence[Mode] = None, upwind_modes: Sequence[Mode] = None):
        """

        Parameters
        ----------
        downwind_modes: Sequence[Mode]
            The downwind Vertical Atmospheric (VA) modes.
        upwind_modes: Sequence[Mode]
            The upwind Vertical Atmospheric (VA) modes.

        """

        self.downwind_modes = downwind_modes    # Acoustic Modes (Type: Array of Mode objects)
        self.upwind_modes = upwind_modes        # Left-Moving Acoustic Modes

    def value_2d(self, x: float, z: float) -> complex:
        """
        Evaluates the acoustic field at (x, z).

        Parameters
        ----------
        x: float
            X-coordinate (range, in :math:`m`).
        z: float
            Z-coordinate (height, in :math:`m`).

        Returns
        -------
        The complex-valued amplitude of the acoustic field at (x, z).
        """

        if x >= 0:
            modes = self.downwind_modes
        else:
            modes = self.upwind_modes

        mode_contributions = np.array([mode.amplitude * mode.value_2d(x, z) for mode in modes])
        value = np.sum(mode_contributions, axis=0)
        return value

    def value_2d_with_3d_spread(self, x: float, z: float, x_ref: float):
        r"""
        Evaluates the acoustic field at (x, z) with spherical spreading accounted for
        with a correction factor.

        Parameters
        ----------
        x: float
            X-coordinate (range, in :math:`m`).
        z: float
            Z-coordinate (height, in :math:`m`).
        x_ref: float
            Reference range (in :math:`m`).

        Returns
        -------
        The complex-valued amplitude of the acoustic field at (x, z) with spherical spreading accounted for.

        Notes
        -----
        The correction factor is given by: :math:`(\sqrt{x\div x_{ref}})^{-1}`, where :math:`x` is the
        range coordinate (in :math:`m`) at which the acoustic field is being evaluated and :math:`x_{ref}` is
        a reference range (also in :math:`m`).
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
        Saves either the downwind or upwind VA modes into a *.txt* file (side='downwind' saves the downwind modes
        and side='upwind' saves the upwind modes).

        Parameters
        ----------
        side: str, optional
            Which set of modes to save (side='downwind' saves the downwind modes and
            side='upwind' saves the upwind modes).


        Returns
        -------
        None
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
        Loads the VA modes from a *.txt* file. Setting side='downwind' loads the downwind VA modes from the file and
        side='upwind' loads the upwind VA modes.

        Parameters
        ----------
        mode_path: str
            Path to the *.txt* file with the VA modes.
        side: str, optional
            Selects the set of modes to be loaded (side='downwind' loads the downwind VA modes
            and side='upwind' the upwind ones.

        Returns
        -------
        None
        """

        data = np.loadtxt(mode_path, dtype=complex)
        print("shape_of_sata", np.shape(data))
        n_modes = len(data[0, :])-1
        heights = np.real(data[0:-3, 0]).ravel()
        modes = []
        for i in range(1, n_modes+1):
            eigvec = data[0:-3, i].ravel()
            ref_wn = data[-3, i]
            eigval = data[-2, i]
            amplitude = data[-1, i]

            modes.append(Mode(complex(amplitude), heights, complex(eigval), eigvec, complex(ref_wn)))

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
            A list of ranges (in :math:`m`) at which the acoustic field is evaluated.
        side: str, optional
            The set of modes used to compute the plot data (side='downwind' for downwind modes and
            side='upwind' for the upwind modes)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing matrices for the range (in :math:`m`),
            height (in :math:`m`) and complex-valued acoustic pressure amplitude at each grid point.
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
            eigvec = mode.amplitude * np.array(mode.eigvec)
            range_shift = np.exp(-1j*mode.eigval*mode.ref_wn*np.array(rangos)) * np.ones((len(z_coords), len(rangos)))
            p = p + eigvec.reshape((len(eigvec), 1)) * range_shift
        return x, z, p

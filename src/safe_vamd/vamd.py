import numpy as np
import scipy.linalg as linalg
from typing import Sequence
from numpy.typing import NDArray
from time import perf_counter

from safe_vamd.solver.acoustic_field import AcousticField
from safe_vamd.solver.mode_filter import ModeFilter
from safe_vamd.solver.mode import Mode


class VAMD:
    """
    Represents an application of the Vertical-Atmospheric Mode-Decomposition (VAMD) method where a set of acoustic
    pressure samples (or transfer functions) is decomposed onto a set of vertical atmospheric (VA) acoustic modes.

    Attributes
    ----------
    acoustic_field : AcousticField
        An instance of a AcousticField with the VA acoustic modes.
    mode_filter : ModeFilter
        An instance of a ModeFilter with methods capable of filtering the original VA modal basis.
    x_samples : Sequence[float]
        The x-coordinates (range) for the samples taken in the Source-Receiver plane.
    z_samples : Sequence[float]
        The z-coordinates (height) for the samples taken in the Source-Receiver plane.
    p-samples : Sequence[complex]
        The complex-valued acoustic pressure (or transfer function) samples taken in the Source-Receiver plane.

    Parameters for initialization.
    Parameters
    ----------
    acoustic_field : AcousticField, optional
        An instance of a AcousticField with the VA acoustic mode basis.
    x_samples : Sequence[float], optional
        The x-coordinates (range) for the samples taken in the Source-Receiver plane.
    z_samples : Sequence[float], optional
        The z-coordinates (height) for the samples taken in the Source-Receiver plane.
    p-samples : Sequence[complex], optional
        The complex-valued acoustic pressure (or transfer function) samples taken in the Source-Receiver plane.
    """
    def __init__(self, acoustic_field: AcousticField = None, x_samples: Sequence[float] = None,
                 z_samples: Sequence[float] = None, p_samples: Sequence[complex] = None) -> None:

        self.acoustic_field = acoustic_field
        self.mode_filter = ModeFilter(acoustic_field)
        self.x_samples = x_samples
        self.z_samples = z_samples
        self.p_samples = p_samples

    def load_samples_from_txt(self, sample_path: str) -> None:
        """
        Loads the acoustic pressure samples from a semicolon separated .txt file. with columns for, respectively,
        height, range, real part and imaginary part of the acoustic pressure samples (the .txt file contains a header).

        Parameters
        ----------
        param sample_path: str
            The path to the txt file containing the acoustic pressure samples.

        Returns
        -------
        None
        """

        time0 = perf_counter()
        data = np.loadtxt(sample_path, skiprows=1, dtype=float, delimiter=';')
        self.x_samples = data[:, 0].ravel()
        self.z_samples = data[:, 1].ravel()
        self.p_samples = data[:, 2].ravel() + 1j * data[:, 3].ravel()
        time1 = perf_counter()
        print("Loading samples took: ", time1 - time0, 's')

    def save_modes_in_txt(self) -> None:
        """
        saves the present mode basis in .txt files with
         the save_the_modes_in_txt method in the acoustic field attribute.

        For N number of modes with eigenvectors of size K, the .txt file has the following format:

        height[0]       eigenvector_mode_1[0]       ...     eigenvector_mode_N[0]
        ...             ...                         ...     ...
        height[K]       eigenvector_mode_1[K]       ...     eigenvector_mode_N[K]
        ###             amplitude_mode_1            ...     amplitude_mode_N
        ###             eigenvalue_mode_1           ...     eigenvalue_mode_N
        ###             ref_wavenumber_mode_1       ...     ref_wavenumber_mode_N

        Returns
        -------
        None
        """

        if self.acoustic_field.downwind_modes is None and self.acoustic_field.upwind_modes is None:

            print('No acoustic modes are loaded!')
            return

        elif self.acoustic_field.downwind_modes is None and self.acoustic_field.upwind_modes is not None:

            print('Saving only the upwind modes')
            self.acoustic_field.save_modes_in_txt(side='upwind')
            return

        elif self.acoustic_field.downwind_modes is not None and self.acoustic_field.upwind_modes is None:

            print('Saving only the downwind modes')
            self.acoustic_field.save_modes_in_txt(side='downwind')
            return

        elif self.acoustic_field.downwind_modes is not None and self.acoustic_field.upwind_modes is not None:

            print('Saving the downwind and upwind modes')
            self.acoustic_field.save_modes_in_txt(side='downwind')
            self.acoustic_field.save_modes_in_txt(side='upwind')
            return

    def load_modes_from_txt(self, downwind_path: str = None, upwind_path: str = None) -> None:
        """
        Receives the path to text file(s) with the downwind, upwind or both sets of modes, creates an instance of
        AcousticField and uses its method (.load_modes_from_txt) to load the acoustic modes. It then sets it as an
        attribute of the VAMD class.

        For a mode basis with eigenvectors of size N and K number of VA modes, the .txt file should
         have the following format:

        height[0]       eigenvector_mode_1[0]       ...     eigenvector_mode_N[0]
        ...             ...                         ...     ...
        height[K]       eigenvector_mode_1[K]       ...     eigenvector_mode_N[K]
        ###             amplitude_mode_1            ...     amplitude_mode_N
        ###             eigenvalue_mode_1           ...     eigenvalue_mode_N
        ###             ref_wavenumber_mode_1       ...     ref_wavenumber_mode_N

        Parameters
        ----------
        downwind_path : str, optional
            The path to the text file with the downwind VA modes.
        upwind_path : str, optional
            The path to the text file with the upwind VA modes.
        Returns
        -------
        None
        """

        time0 = perf_counter()
        acoustic_field = AcousticField()

        if downwind_path is None and upwind_path is None:
            print('Specificy at least one path!')
            return
        elif downwind_path is None and upwind_path is not None:
            acoustic_field.load_modes_from_txt(upwind_path, side='upwind')
            print("Acoustic field with just upwind modes loaded!")

        elif downwind_path is not None and upwind_path is None:
            acoustic_field.load_modes_from_txt(downwind_path, side='downwind')
            print("Acoustic field with just downwind modes loaded!")

        elif downwind_path is not None and upwind_path is not None:
            acoustic_field.load_modes_from_txt(downwind_path, side='downwind')
            acoustic_field.load_modes_from_txt(upwind_path, side='upwind')
            print("Acoustic field with downwind and upwind modes loaded!")

        self.mode_filter.acoustic_field = acoustic_field
        self.acoustic_field = acoustic_field
        time1 = perf_counter()
        print("Loading the modes took: ", time1 - time0, 's')

    def match_modes_to_samples(self, side: str = "downwind", spread_3d: bool = False) -> AcousticField:
        """
        Sets a regression problem (least-squares) for the sample residuals between the samples and VA mode basis
        and solves it with a pseudo-inverse (see Section 2.4 in https://doi.org/10.1016/j.enganabound.2025.106308).
        The problem is set up on an auxiliary coordinate system  closer to the samples (at x_ref) is defined close to
        the samples to improve conditioning. The modal amplitudes computed at this referential are then converted
        into the original coordinate system (x=0) and the individual mode amplitudes updated.

        Parameters
        ----------
        side : str, optional
            Which set of modes should be used for the mode decomposition (side = "downwind" sets the procedure use the
         downwind modal set, "upwind" sets it to use the upwind modes).
        spread_3d: bool, optional
            Toggles the use of a correction factor accounting for spherical spreading when decomposing into the modes.
        Returns
        -------
        None
        """

        time0 = perf_counter()
        if side == "downwind":
            x_ref = np.min(self.x_samples)
            modes = self.acoustic_field.downwind_modes
        else:
            x_ref = np.max(self.x_samples)
            modes = self.acoustic_field.upwind_modes

        pseudo_m = self.compute_pseudo_m(modes, x_ref=x_ref, spread_3d=spread_3d)
        modal_amplitudes_at_cloud = np.matmul(pseudo_m, self.p_samples)
        amp_factors = self.calculate_amp_factors(modes, x_ref=x_ref)
        modal_amplitudes = np.multiply(modal_amplitudes_at_cloud, amp_factors)

        for i in range(0, len(modes)):
            modes[i].amplitude = modal_amplitudes[i]
        time1 = perf_counter()
        print("Matching modes to samples took: ", time1 - time0, 's')
        return self.acoustic_field

    def compute_matrix_m(self, modes: Sequence[Mode], x_ref: float = 0) -> NDArray:
        """
        The individual contributions of the VA modes on the sampling positions, expressed in matrix form
        (see Eq. 15 in https://doi.org/10.1016/j.enganabound.2025.106308).

        Parameters
        ----------
        modes: Sequence[Mode]
            List of modes (instances of Mode) for which the compute matrix is computed.
        x_ref: float, optional
            Range reference where the decay of the modes is set to zero.

        Returns
        -------
        A two-dimensional numpy array with the modal contributions (second-dimension/columns) at each sampling location
         (first-dimension/rows).
        """

        n_samples = len(self.x_samples)

        xs = np.array(self.x_samples).ravel() - x_ref
        zs = np.array(self.z_samples).ravel()
        k0s = np.array([mode.ref_wn for mode in modes])
        eigs = np.array([mode.eigval for mode in modes])
        kxs = np.multiply(eigs, k0s)
        n_modes = len(eigs)

        # Building matrix A = [[e^(-ik0λ1*x1), ..., e^(-ik0λM*x1)], ..., [e^(-ik0λ1*xN), ..., e^(-ik0λM*xN)]]
        a = np.exp(-1j * np.multiply(kxs, np.reshape(xs, [n_samples, 1])))

        # Building matrix B = [[p1(z1), ..., pM(z1)], ..., [p1(zN), ..., pM(zN)]]
        b = np.zeros((n_samples, n_modes), dtype=complex)

        for i in range(0, n_samples):
            b[i, :] = np.array([mode.value(zs[i]) / mode.amplitude for mode in modes])

        sampling_matrix = np.multiply(a, b)

        return sampling_matrix

    def compute_pseudo_m(self, modes, x_ref=0, spread_3d: bool = False) -> np.ndarray:
        """
        Computes the pseudo-inverse matrix to get the modal amplitudes from the samples.

        Parameters
        ----------
        modes: Sequence[Mode]
            List of modes (instance of Mode) for which the pseudo-inverse matrix is computed.
        x_ref: float, optional
            Reference range coordinate.
        spread_3d: bool, optional
            Toggles the use of a correction factor accounting for spherical spreading.
        Returns
        -------
        np.ndarray
            The computed Moore-Penrose pseudo-inverse matrix
        """

        matrix_m = self.compute_matrix_m(modes, x_ref=x_ref)

        if spread_3d:
            n_samples = len(self.x_samples)
            spread_factors = (1 / np.sqrt(self.x_samples / x_ref))
            spread_matrix = np.zeros((n_samples, n_samples), dtype=complex)
            np.fill_diagonal(spread_matrix, spread_factors)
            matrix_m = np.matmul(spread_matrix, matrix_m)

        pseudo_m = linalg.pinv(matrix_m)
        return pseudo_m

    def calculate_amp_factors(self, modes: Sequence[Mode], x_ref: float = 0) -> Sequence[complex]:

        """
        Computes the convertion factors for the mode amplitudes.

        Parameters
        ----------
        modes: Sequence[Mode]
            List of modes (instance of Mode) for which the convertion factors are computed.
        x_ref: float, optional
            Reference range coordinate.
        Returns
        -------
        Sequence[complex]
            The convertion factors from the auxiliary referential (at x_ref) to the original one.
        """

        k0s = np.array([mode.ref_wn for mode in modes])
        eigs = np.array([mode.eigval for mode in modes])
        kxs = np.multiply(eigs, k0s)

        return np.exp(1j * kxs * x_ref)

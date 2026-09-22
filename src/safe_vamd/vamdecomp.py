import numpy as np
import scipy.linalg as linalg
from typing import Sequence
from numpy.typing import NDArray
from time import perf_counter

from safe_vamd.solver.acoustic_field import AcousticField
from safe_vamd.solver.mode_filter import ModeFilter
from safe_vamd.solver.mode import Mode


class VAMDecomp:
    """
    Represents an application of Vertical-Atmospheric Mode-Decomposition (VAMD).

    **Class attributes:**

    Attributes
    ----------
    acoustic_field : AcousticField
         An instance of *AcousticField* with the Vertical Atmospheric (VA) mode basis.
    mode_filter : ModeFilter
        An instance of *ModeFilter* with methods capable of filtering the original VA mode basis.
    x_samples : Sequence[float]
        The x-coordinates (range, in :math:`m`) for the samples taken in the Source-Receiver plane.
    z_samples : Sequence[float]
        The z-coordinates (height, in :math:`m`) for the samples taken in the Source-Receiver plane.
    p-samples : Sequence[complex]
        The complex-valued acoustic pressures (in :math:`Pa`) or transfer functions (unitless)
        sampled in the Source-Receiver plane.

    """
    def __init__(self, acoustic_field: AcousticField = None, x_samples: Sequence[float] = None,
                 z_samples: Sequence[float] = None, p_samples: Sequence[complex] = None) -> None:
        """

        Parameters
        ----------
        acoustic_field : AcousticField, optional
             An instance of *AcousticField* with the Vertical Atmospheric (VA) mode basis.
        x_samples : Sequence[float], optional
            The x-coordinates (range, in :math:`m`) for the samples taken in the Source-Receiver plane.
        z_samples : Sequence[float], optional
            The z-coordinates (height, in :math:`m`) for the samples taken in the Source-Receiver plane.
        p-samples : Sequence[complex], optional
            The complex-valued acoustic pressures (in :math:`Pa`) or transfer functions (unitless)
            sampled in the Source-Receiver plane.
        """

        self.acoustic_field = acoustic_field
        self.mode_filter = ModeFilter(acoustic_field)
        self.x_samples = x_samples
        self.z_samples = z_samples
        self.p_samples = p_samples

    def load_samples_from_txt(self, sample_path: str) -> None:
        """
        Loads the acoustic samples from a semicolon separated *.txt* file with a header followed by
        columns for, respectively, height (in :math:`m`), range (in :math:`m`), real part (in :math:`Pa`, or unitless)
        and imaginary part (in :math:`Pa`, or unitless).

        Parameters
        ----------
        sample_path: str
            The path to the *.txt* file containing the acoustic samples (pressure or transfer functions).

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

    def load_modes_from_txt(self, downwind_path: str = None, upwind_path: str = None) -> None:
        r"""
        Receives the path to *.txt* file(s) with the downwind, upwind or both set(s) of modes, creates an instance of
        *AcousticField*, defines it as an attribute of *VAMDecomp* and uses its *load_modes_from_txt* method
        to load the VA modes.

        For a mode basis with eigenvectors of size :math:`N` and :math:`K` number of VA modes,
        the *.txt* file for each mode basis should have the following format:

        .. math::

           \begin{matrix}
              z_1 & p_1(z_1) & \dots & p_K(z_1) \\
              \dots & \dots & \dots & \dots \\
              z_N & p_1(z_N) & \dots & p_K(z_N) \\
              \hline
              \text{#} & A_1 & \dots & A_K \\
              \text{#} & \lambda_1 & \dots & \lambda_K \\
              \text{#} & k^{ref}_1 & \dots & k^{ref}_K
           \end{matrix}

        , where :math:`p_i(z_j)` represents the shape at height :math:`(z_j)`, :math:`A_i`
        the amplitude,
        :math:`\lambda_i` the eigenvalue, and :math:`k^{ref}_i` the reference wavenumber for mode i.
        The symbol # represents irrelevant information in the *.txt* file.

        Parameters
        ----------
        downwind_path : str, optional
            The path to the *.txt* file with the downwind VA modes.
        upwind_path : str, optional
            The path to the *.txt* file with the upwind VA modes.
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

    def match_modes_to_samples(self, side: str = "downwind", spread_3d: bool = False,
                               condition_monitor: bool = False) -> AcousticField | tuple[AcousticField, float]:
        """
        Sets a regression problem (:math:`P \simeq M^{'} \cdot A^{'}`) between the acoustic samples :math:`P` and a
        linear combination of the VA mode basis, represented by :math:`M^{'} \cdot A^{'}`,
        and computes the least-squares solution (see Section 2.4 in https://doi.org/10.1016/j.enganabound.2025.106308).
        The problem is set up on an auxiliary coordinate system :math:`(x' = x - x_{\\text{ref}}, z' = z)`, best
        defined near the sampling positions :math:`(x_j, z_j)` to improve conditioning.
        The auxiliary coord. system has its origin at :math:`x_{ref}`. The modal amplitudes computed
        at the auxiliary referential are then converted
        into the original coordinate system (:math:`A = E \cdot A^{'}`).

        Parameters
        ----------
        side : str, optional
            Which mode basis should be used in the mode decomposition (side = "downwind" makes the procedure use
            the downwind mode basis, side="upwind" sets it to use the upwind modes).
        spread_3d: bool, optional
            Toggles the use of a correction factor accounting for spherical spreading when decomposing into the modes.
        condition_monitor: bool, optional
            Toggles the computation (and return) of a condition number for the regression problem.
        Returns
        -------
        AcousticField, tuple[AcousticField, float]
            Returns only the acoustic field resultant from the mode decomposition or, additionally, the condition number
            of the pseudo-inverse.
        """

        time0 = perf_counter()
        if side == "downwind":
            x_ref = np.min(self.x_samples)
            modes = self.acoustic_field.downwind_modes
        else:
            x_ref = np.max(self.x_samples)
            modes = self.acoustic_field.upwind_modes

        pseudo_m = self.compute_pseudo_m(modes, x_ref=x_ref, spread_3d=spread_3d)
        modal_amplitudes_at_aux = np.matmul(pseudo_m, self.p_samples)
        matrix_e = self.compute_conv_factors(modes, x_ref=x_ref)
        modal_amplitudes = np.matmul(matrix_e, modal_amplitudes_at_aux)

        for i in range(0, len(modes)):
            modes[i].amplitude = modal_amplitudes[i]
        time1 = perf_counter()
        print("Matching modes to samples took: ", time1 - time0, 's')

        if condition_monitor:
            condition_number = np.linalg.cond(pseudo_m)
            return self.acoustic_field, condition_number
        else:
            return self.acoustic_field

    def compute_matrix_m(self, modes: Sequence[Mode], x_ref: float = 0) -> NDArray:
        """
        The individual contributions of the VA modes on the sampling positions, expressed in matrix form.

        The contributions are computed in the auxiliary coordinate system :math:`(x' = x - x_{\\text{ref}}, z' = z)`,
        best defined near the sampling positions :math:`(x_j, z_j)`, to improve conditioning.

        For :math:`S` number of samples and :math:`K` number of modes, the method assembles the following matrix:

        .. math::
            M' = \\begin{bmatrix}
                    p_1(z'_1) \\cdot e^{-ik_1 x'_1} & \\dots & p_K(z'_1) \\cdot e^{-ik_K x'_1} \\\\
                    \\vdots & \\ddots & \\vdots \\\\
                    p_1(z'_S) \\cdot e^{-ik_1 x'_S} & \\dots & p_K(z'_S) \\cdot e^{-ik_K x'_S}
                 \\end{bmatrix}

        where :math:`p_i(z_j)` and :math:`k_i` are the value of the vertical shape and wave number, respectively,
        for mode :math:`i`.

        Parameters
        ----------
        modes : Sequence[Mode]
            List of VA modes for which matrix :math:`M'` is computed.
        x_ref : float, optional
            Range origin for the auxiliary coordinate system.

        Returns
        -------
        NDArray
            Matrix :math:`M'_{S \\times K}` – a two-dimensional numpy array with the contributions of :math:`K`
            number of VA modes at :math:`S` number of sampling locations.
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
        r"""
        Computes the pseudo-inverse of matrix :math:`M^{'}` to solve the regression problem (least-squares)
        :math:`P \simeq M^{'} \cdot A^{'}`
        , where :math:`S` number of acoustic samples, expressed by matrix :math:`P`,
        are approximated by a linear combination of :math:`K` number of VA modes,
        expressed in matrix form by :math:`M^{'}\cdot A^{'}`.

        Parameters
        ----------
        modes: Sequence[Mode]
            List of modes (instances of *Mode*) for which the pseudo-inverse matrix is computed.
        x_ref: float, optional
            Range origin for the auxiliary coordinate system.
        spread_3d: bool, optional
            Toggles the use of a correction factor :math:`(\sqrt{x \div x_{ref}})^{-1}`
            accounting for spherical spreading.
        Returns
        -------
        np.ndarray
            Matrix :math:`M^{-1'}_{K \times S}` – the pseudo-inverse of matrix :math:`M^{'}`.
        """

        matrix_m = self.compute_matrix_m(modes, x_ref=x_ref)  # Computing matrix M'

        if spread_3d:
            n_samples = len(self.x_samples)
            spread_factors = (1 / np.sqrt(self.x_samples / x_ref))
            spread_matrix = np.zeros((n_samples, n_samples), dtype=complex)
            np.fill_diagonal(spread_matrix, spread_factors)
            matrix_m = np.matmul(spread_matrix, matrix_m)

        pseudo_m = linalg.pinv(matrix_m)
        return pseudo_m

    def compute_conv_factors(self, modes: Sequence[Mode], x_ref: float = 0) -> Sequence[complex]:

        r"""
        Computes the conversion diagonal matrix :math:`E` for :math:`K` number of
        mode amplitudes (:math:`A = E \cdot A^{'}`).

        .. math::
            E = \begin{bmatrix}
                    e^{ik_1x_{ref}} & 0 & \dots & 0 \\
                    0 & e^{ik_2x_{ref}} & \dots & 0 \\
                    \vdots & \vdots & \ddots & \vdots \\
                    0 & 0 & \dots & e^{ik_2x_{ref}}
                \end{bmatrix}
        , where :math:`k_i` is the wavenumber for mode :math:`i`.

        Parameters
        ----------
        modes: Sequence[Mode]
            List of modes (instances of *Mode*) for which matrix :math:`E` is computed.
        x_ref: float, optional
            Range origin for the auxiliary coordinate system.
        Returns
        -------
        Sequence[complex]
            Matrix :math:`E_{K \times K}` - a diagonal matrix with the conversion factors for the modal amplitudes.
        """

        k0s = np.array([mode.ref_wn for mode in modes])
        eigs = np.array([mode.eigval for mode in modes])
        kxs = np.multiply(eigs, k0s)

        return np.diag(np.exp(1j * kxs * x_ref))

    def save_modes_in_txt(self) -> None:
        r"""
        saves the mode basis in *.txt files* with
        the *save_modes_in_txt* method in the *acoustic_field* attribute.

        For :math:`K` number of VA modes with eigenvectors of size :math:`N`, the *.txt* files have
        the following format:

        .. math::

           \begin{matrix}
              z_1 & p_1(z_1) & \dots & p_K(z_1) \\
              \dots & \dots & \dots & \dots \\
              z_N & p_1(z_N) & \dots & p_K(z_N) \\
              \hline
              \text{#} & A_1 & \dots & A_K \\
              \text{#} & \lambda_1 & \dots & \lambda_K \\
              \text{#} & k^{ref}_1 & \dots & k^{ref}_K
           \end{matrix}

        , where :math:`p_i(z_j)` represents the vertical shape at height :math:`(z_j)`, :math:`A_i`
        mode amplitude,
        :math:`\lambda_i` the eigenvalue, and :math:`k^{ref}_i` the reference wavenumber for mode i.
        The symbol # represents irrelevant information in the *.txt* file.

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
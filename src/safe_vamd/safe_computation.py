
import math
import numpy as np
from time import perf_counter

from safe_vamd.solver.acoustic_field import AcousticField
from safe_vamd.solver.kirby_solver_lagrange import kirby_solver_lagrange

from safe_vamd.source.source import Source

from safe_vamd.background_atmospheric_field.air_properties import AirProperties
from safe_vamd.background_atmospheric_field.background_field import BackgroundField
from safe_vamd.stratified_atmosphere.speed_of_sound_stratified_atmosphere import CStratAtmo
from safe_vamd.stratified_atmosphere.pressure_stratified_atmosphere import PStratAtmo
from safe_vamd.stratified_atmosphere.density_stratified_atmosphere import RhoStratAtmo
from safe_vamd.mathematical_functions.pchip_interpolated_func import PchipInterpolator1D
from safe_vamd.boundary_conditions.ground_ingard_myers_boundary_condition import GroundIngardMyersBC

from safe_vamd.solver.kirby_solver_lagrange import kirby_solver_lagrange

# Mesh
from safe_vamd.mesh.mesh_1d import Mesh1D
from safe_vamd.mesh.kirby_c import KirbyC

# Mathematical Functions
from safe_vamd.mathematical_functions.division_func import DivFunc
from safe_vamd.mathematical_functions.constant_func import ConstFunc
from safe_vamd.mathematical_functions.branched_func import BranchedFunc
from safe_vamd.mathematical_functions.derivative_func import DerivFunc
from safe_vamd.mathematical_functions.power_func import PowerFunc
from safe_vamd.mathematical_functions.polynomial_func import PolyFunc

# Dimensionless Numbers
from safe_vamd.dimensionless_quantities.wave_number_with_absorption import WaveNumberWithAbsorption
from safe_vamd.dimensionless_quantities.mach_number import MachNumber
from safe_vamd.dimensionless_quantities.wave_number import WaveNumber


class SAFEComputation:
    """
    A class that represents an application of the Semi-Analytic Finite-Element (SAFE) method.

    **Class attributes:**

    Attributes
    ----------
    source : Source
        An instance of *Source* with information about the acoustic source.
    backgrnd_field : BackgroundField
        An instance of *BackgroundField* representing the atmospheric background field.
    mesh : 'Mesh1D'
        An instance of *Mesh1D* with the elements and nodes.
    acoustic_field : AcousticField
        An instance of *AcousticField* with the Vertical Atmospheric (VA) mode basis.
    alpha : MathFunc
        A damping coefficient (in dB/wavelength), implemented as a *MathFunc*
        to set added absorption in the Perfectly Matched Layer.
    air_absorv : bool
        A boolean to toggle air absorption (air_absorv = True makes the simulation take air absorption into account).
    ground_impedance_value : complex
        The complex-valued normalized surface ground impedance.

    """
    def __init__(self, source: Source = None, backgrnd_field: BackgroundField = None, mesh: Mesh1D = None,
                 acoustic_field: AcousticField = None, alpha_value: float = 0.2, air_absorv: bool = False,
                 ground_impedance_value: complex = None):
        """

        Parameters
        ----------
        source : Source, optional
            An instance of *Source* with monopole source information.
        backgrnd_field : BackgroundField, optional
            An instance of *BackgroundField* representing the atmospheric background field.
        mesh : Mesh1D, optional
            An instance of *Mesh1D* with the finite-elements and nodes.
        acoustic_field : AcousticField, optional
            An instance of *AcousticField* with the Vertical Atmospheric (VA) mode basis.
        alpha_value : float, optional:
            A damping coefficient (in dB/wavelength) to set added absorption in the Perfectly Matched Layer.
        air_absorv : bool, optional
            A boolean to toggle air absorption (air_absorv = True) makes the computation take air absorption
            into account.
        ground_impedance_value : complex, optional
            The complex-valued normalized surface ground impedance.

        """

        self.mesh = mesh
        self.source = source
        self.backgrnd_field = backgrnd_field
        self.acoustic_field = acoustic_field

        self.alpha = ConstFunc(alpha_value)  # Added attenuation in the PML [dB/λ]
        self.air_absorv = air_absorv  # True if the acoustic field was computed taking into account air absorption.
        self.ground_impedance_value = ground_impedance_value

    def setup(self, input_file: str, f: float, rho_at_ground: float, g_value: float = 9.81,
              pml_thick_factor: float = 2) -> None:
        """
        Builds the objects related to the computational mesh and background atmospheric field from an inputted
        *.txt* file which has a header followed by columns for height (in :math:`m`), temperature (in :math:`K`),
        wind-velocity (in :math:`ms^{-1}`) and relative humidity (in :math:`\%`).

        Parameters
        ----------
        input_file: str
            Path to the input *.txt* file with the atmospheric conditions.
        f: float
            Excitation frequency (in :math:`Hz`)
        rho_at_ground: float
            Value of the air density (in :math:`kgm^{-3}`) at ground level.
        g_value: float, optional
            Value for the acceleration of gravity (in :math:`ms^{-2}`).
        pml_thick_factor: float, optional
            A unitless parameter that controls the thickness of the Perfectly Matched in proportion to
            the maximum wavelength.

        Returns
        -------
        None
        """

        time0 = perf_counter()
        # Flushing variables in the SAFE computation
        self.source = None
        self.backgrnd_field = None
        self.acoustic_field = None
        self.alpha = None
        self.air_absorv = None

        # Load the input file and perform checks
        data = np.loadtxt(input_file, dtype=float, skiprows=1, delimiter=';')
        zs = data[:, 0].ravel()
        t0s = data[:, 1].ravel()
        u0s = data[:, 2].ravel()
        rhs = data[:, 3].ravel()
        print("Number of data points: ", len(zs))

        # Doing checks on the data (Correct number of nodes)
        if (int(len(zs)) - 1) % 2 != 0:
            raise ValueError('Number of nodes should be 2*N+1, where N is the number of quadratic elements!')

        # Doing checks on the data (first coordinate should be zero)
        if zs[0] != 0:
            print('First height coordinate is not at ground level (z=0)! Zeroing height data w.r.t :', np.min(zs))
            zs = zs - np.min(zs)

        n_inner_elements = (int(len(zs)) - 1) / 2  # considering quadratic elements

        # Creating Source object
        source = Source(2 * np.pi * f, 0, 0)

        # Making sure the data is ordered from lowest to greatest height
        indexes = np.argsort(zs)
        zs = zs[indexes]
        t0s = t0s[indexes]
        u0s = u0s[indexes]
        rhs = rhs[indexes]

        # Computing PML (Perfectly Matched Layer) thickness (2 times the greatest wave length)
        inner_domain_height = np.max(zs) - np.min(zs)
        greatest_wavelength = np.sqrt(1.4 * 287 * (273.15 + np.max(t0s))) / f
        pml_thickness = math.ceil(pml_thick_factor * greatest_wavelength)
        n_pml_elements = math.ceil((pml_thickness/inner_domain_height) * n_inner_elements)

        # Setting up a mesh (Piecewise Quadratic Elements)
        mesh = Mesh1D()
        element_boundaries = zs[0::2]  # making sure the inner points in the elements are equally spaced
        mesh.load_from_element_bounds(element_boundaries, degree=2)
        mesh.add_top_pml(pml_thickness, n_pml_elements, degree=2)
        print("Total number of elements:", len(np.array(mesh.elements)))
        print("Number of PML elements:", n_pml_elements)
        print("Thickness of PML:", pml_thickness, ' m')

        # Setting up the Background Atmospheric Field
        air_properties = AirProperties(ConstFunc(1.4), ConstFunc(287))

        t0 = PchipInterpolator1D(zs, t0s)   # Air temperature field
        t0 = BranchedFunc(t0, ConstFunc(t0.value(inner_domain_height)), inner_domain_height)  # constant temperature
        # in the PML

        rh = PchipInterpolator1D(zs, rhs)  # Relative Humidity
        rh = BranchedFunc(rh, ConstFunc(rh.value(inner_domain_height)), inner_domain_height)  # constant relative
        # humidity in the PML

        vx0 = PchipInterpolator1D(zs, u0s)  # Longitudinal Wind speed
        vx0 = BranchedFunc(vx0, ConstFunc(vx0.value(inner_domain_height)), inner_domain_height)  # constant wind speed
        # in the PML

        vy0 = ConstFunc(0)                  # Cross Wind (not considered in SAFE)
        vz0 = ConstFunc(0)                  # Vertical Wind (not considered in SAFE)

        g = ConstFunc(g_value)

        rho0 = RhoStratAtmo(0, g, rho_at_ground, t0, air_properties)   # Density (Hydrostatic relations) [Kg/m^3]
        p0 = PStratAtmo(rho0, t0, air_properties)           # Pressure Field (Perfect Gas Law) [Pa]
        c = CStratAtmo(t0, air_properties)          # Speed of Sound Field [m/s]

        # Computing the derived atmospheric conditions at the input heights
        rho0_values = np.array([rho0.value(z) for z in zs]).ravel()
        p0_values = np.array([p0.value(z) for z in zs]).ravel()
        c_values = np.array([c.value(z) for z in zs]).ravel()
        
        rho0 = PchipInterpolator1D(zs, rho0_values)  # Density (PCHIP cubic spline)
        rho0 = BranchedFunc(rho0, ConstFunc(rho0.value(inner_domain_height)), inner_domain_height)  # const. value
        # in PML

        p0 = PchipInterpolator1D(zs, p0_values)  # Absolute Pressure (PCHIP cubic spline)
        p0 = BranchedFunc(p0, ConstFunc(p0.value(inner_domain_height)), inner_domain_height)  # const. value
        # in PML

        c = PchipInterpolator1D(zs, c_values)  # Speed of Sound (PCHIP cubic spline)
        c = BranchedFunc(c, ConstFunc(c.value(inner_domain_height)), inner_domain_height)  # const. value in PML

        backfield = BackgroundField(air_properties, g, rho0, rh, p0, t0, vx0, vy0, vz0, c)

        self.mesh = mesh
        self.source = source
        self.backgrnd_field = backfield
        self.acoustic_field = None
        time1 = perf_counter()
        print('Setting up took:', time1 - time0, 's')

    def solve(self, ground_impedance_value: complex = None, alpha_value: float = 0.2,
              air_absorption: bool = False) -> AcousticField:
        """
        Assembles the matrices for the cubic eigenvalue problem (:math:'A + B\lambda + C\lambda^2 +D\lambda^4')
        and solves it by using the function kirby_solver_lagrange.

        Parameters
        ----------
        alpha_value: float
            A parameter to set added absorption in the Perfectly Matched Layer.
        air_absorption: bool
            A boolean to toggle air absorption (air_absorv = True makes the simulation take into account air absorption)
            .
        ground_impedance_value: complex
            The complex-valued normalized surface ground impedance.

        Returns
        -------
        AcousticField
            The acoustic pressure field with the upwind and downwind modes.
        """

        time0 = perf_counter()
        source = self.source
        mesh = self.mesh
        backfield = self.backgrnd_field
        alpha = ConstFunc(alpha_value)
        if ground_impedance_value is None:
            field = kirby_solver_lagrange(mesh, backfield, source, alpha, absor=air_absorption)
        else:
            bc = GroundIngardMyersBC(source, backfield, ground_impedance_value)
            field = kirby_solver_lagrange(mesh, backfield, source, alpha, absor=air_absorption, boundary_condition=bc)
        del self.acoustic_field
        self.acoustic_field = field
        self.alpha = alpha
        self.air_absorv = air_absorption
        print("Solving took: ", perf_counter() - time0, "s")

        return self.acoustic_field

    def save_modes_in_txt(self) -> None:
        r"""
        Saves the upwind and downwind modes in separate *.txt* files with
        the *save_modes_in_txt* method in the *acoustic_field* attribute.

        For :math:`K` number of VA modes with eigenvectors of size .math:`N`, the *.txt* file has the following format:

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

        , where :math:`p_i(z_j)` represents the vertical shape at height :math:`z_j`, :math:`A_i`
        the mode amplitude,
        :math:`\lambda_i` the eigenvalue, and :math:`k^{ref}_i` the reference wavenumber for mode i.
        The symbol # represents irrelevant information in the *.txt* file.

        Returns
        -------
        None
        """

        print('Saving the downwind and upwind modes')
        self.acoustic_field.save_modes_in_txt(side='downwind')
        self.acoustic_field.save_modes_in_txt(side='upwind')
        return

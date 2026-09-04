
import math
import numpy as np
from time import perf_counter

from safe_vamd.solver.acoustic_field import AcousticField

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

    Attributes
    ----------
    source: Source
        An instance of the Source class with monopole source information.
    backgrnd_field: BackgroundField
        An instance of the BackgroundField class representing the atmospheric background field.
    mesh: Mesh1D
        An instance of the Mesh1D class with the finite-elements and nodes.
    acoustic_field: AcousticField
        An instance of the AcousticField class with the Vertical Atmospheric (VA) modes
    alpha: MathFunc
        A function to set added absorption in the Perfectly Matched Layer.
    air_absorv: bool
        A boolean to toggle air absorption (air_absorv = True makes the simulation take into account air absorption)
    ground_impedance_value: complex
        The complex-valued normalized surface ground impedance.

    Parameters for initialization.
    Parameters
    ----------
    source: Source, optional
        An instance of the Source class with monopole source information.
    backgrnd_field: BackgroundField, optional
        An instance of the BackgroundField class representing the atmospheric background field.
    mesh: Mesh1D, optional
        An instance of the Mesh1D class with the finite-elements and nodes.
    acoustic_field: AcousticField, optional
        An instance of the AcousticField class with the Vertical Atmospheric (VA) modes
    alpha_value: float, optional:
        A value to set added absorption in the Perfectly Matched Layer.
    air_absorv: bool, optional
        A boolean to toggle air absorption (air_absorv = True makes the simulation take into account air absorption)
    ground_impedance_value: complex
        The complex-valued normalized surface ground impedance.
    """
    def __init__(self, source: Source = None, backgrnd_field: BackgroundField = None, mesh: Mesh1D = None,
                 acoustic_field: AcousticField = None, alpha_value: float = 0.2, air_absorv: bool = False,
                 ground_impedance_value: complex = None):

        self.mesh = mesh
        self.source = source  # Point Source Object (Type: Source)
        self.backgrnd_field = backgrnd_field  # Atmospheric Background Field (Type: BckgrndField)
        self.acoustic_field = acoustic_field    # Acoustic Field

        self.alpha = ConstFunc(alpha_value)  # Added attenuation in the PML [dB/λ]
        self.air_absorv = air_absorv  # True if the acoustic field was computed taking into account air absorption.
        self.ground_impedance_value = ground_impedance_value

    def setup(self, input_file: str, f: float, rho_at_ground: float, g_value: float = 9.81,
              pml_thick_factor: float = 2) -> None:
        """
        Builds the necessary objects related to the computational mesh and background atmospheric field from an inputted
        test file a header and columns for height, temperature, wind-velocity and relative humidity.

        Parameters
        ----------
        input_file: str
            Path to the input test file.
        f: float
            Excitation frequency (in Hz)
        rho_at_ground: float
            Value of the air density at ground level.
        g_value: float, optional
            Value for the acceleration of gravity.
        pml_thick_factor: float, optional
            Parameter that controls the thickness of the Perfectly Matched

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
        return 1

    def set_alpha(self, alpha_value: float):
        self.alpha = ConstFunc(alpha_value)

    def toggle_air_absorv(self, air_absorv: float):
        pass

    def set_ground_impedance(self, ground_impedance_value: complex):
        pass

    def solve(self, ground_impedance_value: complex = None, alpha_value: float = 0.2,
              air_absorption: bool = False) -> AcousticField:
        """
        Assembles the necessary matrices for the cubic eigenvalue problem and solves it by using the function
        kirby_solver_lagrange.

        Parameters
        ----------
        alpha_value: float
            A parameter to set added absorption in the Perfectly Matched Layer.
        air_absorption: bool
            A boolean to toggle air absorption (air_absorv = True makes the simulation take into account air absorption)
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

    def match_to_monopole(self, source_height: float, source_amplitude: float, side: str = 'downwind') -> AcousticField:
        """
        Uses the methodology in https://doi.org/10.1121/10.0002912 to match either the
        upwind or downwind VA modes to a monopole source. it then updates the modal amplitudes on the AcousticField
        object.

        Parameters
        ----------
        source_height: float
            Height (in meters) of the monopole.
        source_amplitude:
            Amplitude (in kg/(m^-3 s)) of the monopole.
        side: str
            Which set of modes should be used for the mode decomposition (side = "downwind" sets the procedure use the
            downwind modal set, "upwind" sets it to use the upwind modes).

        Returns
        -------
        AcousticField
            The acoustic field induced by the monopole.
        """

        # Updating the source
        self.source.amplitude = source_amplitude
        self.source.height = source_height

        background_field = self.backgrnd_field
        mesh = self.mesh
        alpha = self.alpha
        source = self.source
        air_absorv = self.air_absorv

        # Mode objects
        if side == 'downwind':
            modes = self.acoustic_field.downwind_modes
        else:
            modes = self.acoustic_field.upwind_modes

        # Functions
        w = ConstFunc(source.w)                         # Source Frequency [MathFunc]
        cref = ConstFunc(np.real(background_field.c.value(0)))  # Reference speed of sound (ground level) [MathFunc]
        kref = WaveNumber(cref, w)                      # Reference wave number  [MathFunc]
        g = background_field.g                        # Acceleration of Gravity  [MathFunc]
        z_pml = mesh.elements[-1].z_ref                  # Height at which the pml starts  [MathFunc]
        rho0 = background_field.rho                            # Density  [MathFunc]
        u0 = background_field.vx                               # Range Velocity  [MathFunc]
        mach0 = MachNumber(cref, u0)                    # Mach Number  [MathFunc]
        dmach0 = DerivFunc(mach0)                       # Height derivative of mach number  [MathFunc]
        # Modified speed of sound  [MathFunc]
        c = BranchedFunc(background_field.c, KirbyC(background_field.c, alpha), xchange=z_pml)
        gline = DivFunc(g, PowerFunc(c, ConstFunc(2)))  # g/c^2 [MathFunc]
        if air_absorv == 0:
            k = WaveNumber(c, w)
        else:
            k = WaveNumberWithAbsorption(source, background_field, c)

        invxi = DivFunc(ConstFunc(1), mesh.elements[-1].pml)  # 1/xi (inverse of pml stretching function) [MathFunc]

        n_nodes = len(mesh.nodes)
        n_modes = len(modes)
        n_elements = len(mesh.elements)

        # Basic Quantities at the nodes
        zs = np.array([node.z for node in mesh.nodes])
        lambdas = np.reshape(np.array([mode.eigval for mode in modes]), (1, n_modes))
        glines = np.reshape(gline.value(zs), (n_nodes, 1))
        machs = np.reshape(mach0.value(zs), (n_nodes, 1))
        dmachs = np.reshape(dmach0.value(zs), (n_nodes, 1))
        rhos = np.reshape(rho0.value(zs), (n_nodes, 1))
        krefs = np.reshape(kref.value(zs), (n_nodes, 1))
        ks = np.reshape(k.value(zs), (n_nodes, 1))
        invxis = np.reshape(invxi.value(zs), (n_nodes, 1))
        mode_shapes = np.zeros((n_nodes, n_modes), dtype=complex)

        for i in range(0, n_modes):
            mode_shapes[:, i] = modes[i].eigvec
        # integration terms at the nodes
        mode_shapes_squared = np.multiply(mode_shapes, mode_shapes)
        term0 = 1 - np.matmul(machs, lambdas)  # term0 = 1-λM
        term1 = (1/rhos)
        term2 = ((ks ** 2) * machs)/krefs
        term3 = (glines/(krefs * (term0 ** 2))) * invxis * dmachs
        term4 = (krefs * lambdas)/term0
        term5 = (1 + (1j/(krefs*term0)) * invxis * dmachs)
        values_at_nodes = np.multiply(term1 * (term2 - term3 + term4 * term5), mode_shapes_squared)

        # Integration weight from a certain node to a certain element
        glq_weights = np.array([0.555555555555555, 0.888888888888888, 0.555555555555555])
        glq_points = np.array([-0.77459666924148, 0, 0.77459666924148]) + 1

        phi_1_values = np.array(PolyFunc(np.array([0, -0.5, 0.5])).value(glq_points))
        phi_2_values = np.array(PolyFunc(np.array([1, 0, -1])).value(glq_points))
        phi_3_values = np.array(PolyFunc(np.array([0, 0.5, 0.5])).value(glq_points))

        u1 = np.sum(glq_weights * phi_1_values)
        u2 = np.sum(glq_weights * phi_2_values)
        u3 = np.sum(glq_weights * phi_3_values)

        lengths = np.reshape(np.array([element.length for element in mesh.elements]), (1, n_elements))

        integration_weights = np.zeros((n_elements, n_nodes), dtype=complex)
        integration_weights[np.arange(0, n_elements, 1), np.arange(0, n_nodes-2, 2)] = u1 * lengths
        integration_weights[np.arange(0, n_elements, 1), np.arange(1, n_nodes-1, 2)] = u2 * lengths
        integration_weights[np.arange(0, n_elements, 1), np.arange(2, n_nodes, 2)] = u3 * lengths

        amplitudes = np.sum(0.5 * np.matmul(integration_weights, values_at_nodes), axis=0)

        # Final amplitude formula
        rho_value_at_source_height = rho0.value(source.height)
        mode_values_at_source_height = np.array([mode.value(source.height)/mode.amplitude for mode in modes])

        amplitudes = (1j * (source.amplitude/rho_value_at_source_height)
                      * np.divide(mode_values_at_source_height, amplitudes))

        for h in range(0, n_modes):
            modes[h].amplitude = amplitudes[h]
        return self.acoustic_field

    def save_modes_in_txt(self) -> None:
        """
        saves the upwind and downwind modes in separate .txt files with
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

        print('Saving the downwind and upwind modes')
        self.acoustic_field.save_modes_in_txt(side='downwind')
        self.acoustic_field.save_modes_in_txt(side='upwind')
        return


import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
import tracemalloc

from safe_vamd.source.source import Source
from safe_vamd.mesh.mesh_1d import Mesh1D
from safe_vamd.background_atmospheric_field.background_field import BackgroundField
from safe_vamd.boundary_conditions.ground_ingard_myers_boundary_condition import GroundIngardMyersBC

# Math_Funcs
from safe_vamd.mathematical_functions.division_func import DivFunc
from safe_vamd.mathematical_functions.product_func import ProdFunc
from safe_vamd.mathematical_functions.subtraction_func import SubFunc
from safe_vamd.mathematical_functions.summation_func import SumFunc
from safe_vamd.mathematical_functions.constant_func import ConstFunc
from safe_vamd.mathematical_functions.derivative_func import DerivFunc
from safe_vamd.mathematical_functions.power_func import PowerFunc
from safe_vamd.mathematical_functions.interpolated_func import InterpolatedFunc
from safe_vamd.mathematical_functions.cubic_eigen_solver import cubic_eigen_solver
from safe_vamd.mathematical_functions.glq_quadrature_new import glq_quadrature_new

# Mesh
from safe_vamd.mesh.kirby_c import KirbyC
from safe_vamd.mesh.element_1d_arbitrary_order_lagrange import Element1DArbitraryOrderLagrange
from safe_vamd.mesh.element_1d_arbitrary_order_lagrange_pml import Element1DArbitraryOrderLagrangePML

# Dimensionless-Quantities
from safe_vamd.dimensionless_quantities.mach_number import MachNumber
from safe_vamd.dimensionless_quantities.wave_number import WaveNumber
from safe_vamd.dimensionless_quantities.wave_number_with_absorption import WaveNumberWithAbsorption

# Solver
from safe_vamd.solver.acoustic_field import AcousticField
from safe_vamd.solver.mode import Mode


def kirby_solver_lagrange(mesh: Mesh1D, background_field: BackgroundField, source: Source, alpha: float,
                          absor: bool = False, boundary_condition: GroundIngardMyersBC = None) -> AcousticField:
    """
    Assembles the finite-element matrices for the cubic eigenvalue problem (A + Bλ + Cλ^2 + Dλ^3)p = 0, solves it and
    creates an acoustic field with the computed upwind and downwind vertical atmospheric (VA) modes

    Parameters
    ----------
    mesh: Mesh1D
    background_field: BackgroundField
        The atmospheric background fields.
    source: Source
         Information related to a monopole source.
    alpha: float
        A parameter to set added absorption in the Perfectly Matched Layer.
    absor: bool, optional
        Toggles the use of air absorption in the computation of the modes (absor = True makes the computation take into
         consideration air absorption).
    boundary_condition: GroundIngardMyersBC, optional
        An object that applies a IngardMyersBC boundary condition on the ground directly on the cubic
         eigenvalue problem matrices.

    Returns
    -------
    AcousticField
        The acoustic field with the computed upwind and downwind VA modes.
    """

    time0 = perf_counter()

    # Noise Source Frequency (MathFunc Object)
    w = ConstFunc(source.w)

    # Acceleration of Gravity (MathFunc Object)
    g = background_field.g

    # Reference values (MathFunc Objects)
    c0 = ConstFunc(np.real(background_field.c.value(0)))
    k0 = WaveNumber(c0, w)

    # Quantities for PML and Inner Domains (MathFunc Objects)
    rho = background_field.rho
    vx = background_field.vx
    mach = MachNumber(c0, vx)
    dmach = DerivFunc(mach)
    drho = DerivFunc(rho)

    # Inner region (MathFunc Objects)
    c = background_field.c

    if absor:
        k = WaveNumberWithAbsorption(source, background_field, c)
        print("air absorption: ON")
    else:
        k = WaveNumber(c, w)
        print("air absorption: OFF")

    dc = DerivFunc(c)
    gline = DivFunc(g, PowerFunc(c, ConstFunc(2)))

    g1line = ProdFunc([gline, PowerFunc(rho, ConstFunc(-1)), drho])
    g1line = SumFunc(ProdFunc([gline, DivFunc(ConstFunc(2), c), dc]), g1line)
    g1line = SumFunc(ProdFunc([gline, gline]), g1line)

    # Pauls Interpolated Functions (Inner Region)
    rho = InterpolatedFunc(mesh, rho)
    drho = InterpolatedFunc(mesh, drho)
    mach = InterpolatedFunc(mesh, mach)
    dmach = InterpolatedFunc(mesh, dmach)
    k = InterpolatedFunc(mesh, k)
    gline = InterpolatedFunc(mesh, gline)
    g1line = InterpolatedFunc(mesh, g1line)

    # Matrices Functions  (Inner Region) (MathFunc Objects)
    d_f = ProdFunc([mach, SubFunc(ProdFunc([k0, k0]), ProdFunc([k, k, mach, mach]))])
    c_f = SubFunc(ProdFunc([ConstFunc(3), k, k, mach, mach]), ProdFunc([k0, k0]))
    b1_f = mach
    b2_f = SumFunc(ProdFunc([ConstFunc(2), dmach]), ProdFunc([mach, PowerFunc(rho, ConstFunc(-1)), drho]))
    b3_f = ProdFunc([ConstFunc(2), gline, dmach])
    b3_f = SumFunc(ProdFunc([g1line, mach]), b3_f)
    b3_f = SubFunc(b3_f, ProdFunc([ConstFunc(3), k, k, mach]))

    a1_f = SubFunc(g1line, ProdFunc([k, k]))
    a2_f = ProdFunc([PowerFunc(rho, ConstFunc(-1)), drho])
    a3_f = ConstFunc(1)

    # PML Region

    c_pml = KirbyC(background_field.c, alpha)  # modified speed of sound (extra dampening in the PML region)
    k_pml = WaveNumber(c_pml, w)  # wave number with modified speed of sound

    dc_pml = DerivFunc(c_pml)
    gline_pml = DivFunc(g, PowerFunc(c_pml, ConstFunc(2)))
    g1line_pml = ProdFunc([gline_pml, PowerFunc(rho, ConstFunc(-1)), drho])
    g1line_pml = SumFunc(ProdFunc([gline_pml, DivFunc(ConstFunc(2), c), dc_pml]), g1line_pml)
    g1line_pml = SumFunc(ProdFunc([gline_pml, gline_pml]), g1line_pml)

    # Pauls Interpolated Functions (PML Region)
    k_pml = InterpolatedFunc(mesh, k_pml)
    gline_pml = InterpolatedFunc(mesh, gline_pml)
    g1line_pml = InterpolatedFunc(mesh, g1line_pml)

    # Matrix Functions (PML Region)

    d_f_pml = ProdFunc([mach, SubFunc(ProdFunc([k0, k0]), ProdFunc([k_pml, k_pml, mach, mach]))])
    c_f_pml = SubFunc(ProdFunc([ConstFunc(3), k_pml, k_pml, mach, mach]), ProdFunc([k0, k0]))
    b1_f_pml = mach
    b2_f_pml = SumFunc(ProdFunc([ConstFunc(2), dmach]), ProdFunc([mach, PowerFunc(rho, ConstFunc(-1)), drho]))

    b3_f_pml = ProdFunc([g1line_pml, mach])
    b3_f_pml = SubFunc(b3_f_pml, ProdFunc([ConstFunc(3), k_pml, k_pml, mach]))

    b4_f_pml = ProdFunc([ConstFunc(2), gline_pml, dmach])

    a1_f_pml = SubFunc(g1line_pml, ProdFunc([k_pml, k_pml]))
    a2_f_pml = ProdFunc([PowerFunc(rho, ConstFunc(-1)), drho])
    a3_f_pml = ConstFunc(1)

    # Initializing A, B, C and D Matrices
    n_elements = len(mesh.elements)
    npe = len(mesh.elements[0].nodes)
    a = np.zeros(((npe - 1) * n_elements + 1, (npe - 1) * n_elements + 1), dtype='complex_')
    b = np.zeros(((npe - 1) * n_elements + 1, (npe - 1) * n_elements + 1), dtype='complex_')
    c = np.zeros(((npe - 1) * n_elements + 1, (npe - 1) * n_elements + 1), dtype='complex_')
    d = np.zeros(((npe - 1) * n_elements + 1, (npe - 1) * n_elements + 1), dtype='complex_')

    for i in range(0, n_elements, 1):

        # Cell
        cell = mesh.elements[i]

        # Node Identification Numbers (Element i)
        nids = [node.nid for node in mesh.elements[i].nodes]

        # Integration Interval (Element i)
        inter = [mesh.elements[i].nodes[0].z, mesh.elements[i].nodes[-1].z]

        # Basis Functions (Element i)
        bf = mesh.elements[i].bf

        # Derivative of Basis Functions (Element i)
        dbf = mesh.elements[i].dbf

        if type(mesh.elements[i]) is Element1DArbitraryOrderLagrange:  # Inner Domain

            # Assembling the Inner Domain Matrices
            for j in range(0, len(bf)):

                d[nids[j]][nids[j]] = d[nids[j]][nids[j]] + glq_quadrature_new([d_f], [bf[j], bf[j]], inter)

                c[nids[j]][nids[j]] = c[nids[j]][nids[j]] + glq_quadrature_new([c_f], [bf[j], bf[j]], inter)

                b[nids[j]][nids[j]] = b[nids[j]][nids[j]] + glq_quadrature_new([b1_f], [dbf[j], dbf[j]], inter)
                b[nids[j]][nids[j]] = b[nids[j]][nids[j]] + glq_quadrature_new([b2_f], [bf[j], dbf[j]], inter)
                b[nids[j]][nids[j]] = b[nids[j]][nids[j]] + glq_quadrature_new([b3_f], [bf[j], bf[j]], inter)

                a[nids[j]][nids[j]] = a[nids[j]][nids[j]] + glq_quadrature_new([a1_f], [bf[j], bf[j]], inter)
                a[nids[j]][nids[j]] = a[nids[j]][nids[j]] + glq_quadrature_new([a2_f], [bf[j], dbf[j]], inter)
                a[nids[j]][nids[j]] = a[nids[j]][nids[j]] + glq_quadrature_new([a3_f], [dbf[j], dbf[j]], inter)

                ran = list(range(0, len(bf)))
                ran.pop(j)

                for k in ran:

                    d[nids[j]][nids[k]] = d[nids[j]][nids[k]] + glq_quadrature_new([d_f], [bf[j], bf[k]], inter)

                    c[nids[j]][nids[k]] = c[nids[j]][nids[k]] + glq_quadrature_new([c_f], [bf[j], bf[k]], inter)

                    b[nids[j]][nids[k]] = b[nids[j]][nids[k]] + glq_quadrature_new([b1_f], [dbf[j], dbf[k]], inter)
                    b[nids[j]][nids[k]] = b[nids[j]][nids[k]] + glq_quadrature_new([b2_f], [bf[j], dbf[k]], inter)
                    b[nids[j]][nids[k]] = b[nids[j]][nids[k]] + glq_quadrature_new([b3_f], [bf[j], bf[k]], inter)

                    a[nids[j]][nids[k]] = a[nids[j]][nids[k]] + glq_quadrature_new([a1_f], [bf[j], bf[k]], inter)
                    a[nids[j]][nids[k]] = a[nids[j]][nids[k]] + glq_quadrature_new([a2_f], [bf[j], dbf[k]], inter)
                    a[nids[j]][nids[k]] = a[nids[j]][nids[k]] + glq_quadrature_new([a3_f], [dbf[j], dbf[k]], inter)

        elif type(mesh.elements[i]) is Element1DArbitraryOrderLagrangePML:  # PML

            absorption = mesh.elements[i].pml
            invabs = DivFunc(ConstFunc(1), absorption)

            # Assembling the Inner Domain Matrices
            for j in range(0, len(bf)):

                d[nids[j]][nids[j]] = (d[nids[j]][nids[j]] +
                                       glq_quadrature_new([d_f_pml], [absorption, bf[j], bf[j]], inter))

                c[nids[j]][nids[j]] = (c[nids[j]][nids[j]] +
                                       glq_quadrature_new([c_f_pml], [absorption, bf[j], bf[j]], inter))

                b[nids[j]][nids[j]] = (b[nids[j]][nids[j]] +
                                       glq_quadrature_new([b1_f_pml], [invabs, dbf[j], dbf[j]], inter))
                b[nids[j]][nids[j]] = (b[nids[j]][nids[j]] +
                                       glq_quadrature_new([b2_f_pml], [invabs, bf[j], dbf[j]], inter))

                b[nids[j]][nids[j]] = (b[nids[j]][nids[j]] +
                                       glq_quadrature_new([b3_f_pml], [absorption, bf[j], bf[j]], inter))

                b[nids[j]][nids[j]] = (b[nids[j]][nids[j]] +
                                       glq_quadrature_new([b4_f_pml], [bf[j], bf[j]], inter))

                a[nids[j]][nids[j]] = (a[nids[j]][nids[j]] +
                                       glq_quadrature_new([a1_f_pml], [absorption, bf[j], bf[j]], inter))
                a[nids[j]][nids[j]] = (a[nids[j]][nids[j]] +
                                       glq_quadrature_new([a2_f_pml], [invabs, bf[j], dbf[j]], inter))
                a[nids[j]][nids[j]] = (a[nids[j]][nids[j]] +
                                       glq_quadrature_new([a3_f_pml], [invabs, dbf[j], dbf[j]], inter))

                ran = list(range(0, len(bf)))
                ran.pop(j)

                for k in ran:

                    d[nids[j]][nids[k]] = (d[nids[j]][nids[k]] +
                                           glq_quadrature_new([d_f_pml], [absorption, bf[j], bf[k]], inter))

                    c[nids[j]][nids[k]] = (c[nids[j]][nids[k]] + glq_quadrature_new([c_f_pml], [absorption, bf[j], bf[k]], inter))

                    b[nids[j]][nids[k]] = (b[nids[j]][nids[k]] +
                                           glq_quadrature_new([b1_f_pml], [invabs, dbf[j], dbf[k]], inter))
                    b[nids[j]][nids[k]] = (b[nids[j]][nids[k]] +
                                           glq_quadrature_new([b2_f_pml], [invabs, bf[j], dbf[k]], inter))

                    b[nids[j]][nids[k]] = (b[nids[j]][nids[k]] +
                                           glq_quadrature_new([b3_f_pml], [absorption, bf[j], bf[k]], inter))

                    b[nids[j]][nids[k]] = (b[nids[j]][nids[k]] + glq_quadrature_new([b4_f_pml], [bf[j], bf[k]], inter))

                    a[nids[j]][nids[k]] = (a[nids[j]][nids[k]] +
                                           glq_quadrature_new([a1_f_pml], [absorption, bf[j], bf[k]], inter))
                    a[nids[j]][nids[k]] = (a[nids[j]][nids[k]] +
                                           glq_quadrature_new([a2_f_pml], [invabs, bf[j], dbf[k]], inter))
                    a[nids[j]][nids[k]] = (a[nids[j]][nids[k]] +
                                           glq_quadrature_new([a3_f_pml], [invabs, dbf[j], dbf[k]], inter))

    if boundary_condition is not None:
        boundary_condition.apply_bc(a, b, c, d)

    #np.save("example_2_matrix_a", a)
    #np.save("example_2_matrix_b", b)
    #np.save("example_2_matrix_c", c)
    #np.save("example_2_matrix_d", d)

    time1 = perf_counter()
    print("Matrix assembly took:", time1 - time0, 's')

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current matrix creation: {current / 1024 ** 2:.2f} MB")
    print(f"Peak matrix_creation:    {peak / 1024 ** 2:.2f} MB")
    tracemalloc.reset_peak()

    time2 = perf_counter()
    # Solving the eigenvalue problem ( (A - Bγ - Cγ^2 - Dγ^3)p = 0)
    vals, vecs = cubic_eigen_solver(a, b, c, d)
    time3 = perf_counter()
    print("Solving the eigenvalue prob took:", time3 - time2, 's')

    current, peak = tracemalloc.get_traced_memory()
    print(f"Solving eigenvalue problem: {current / 1024 ** 2:.2f} MB")
    print(f"Peak Solving eigenvalue problem:    {peak / 1024 ** 2:.2f} MB")
    tracemalloc.reset_peak()

    time4 = perf_counter()

    # Filtering out spurious modes (high real part)
    abs_imag_values = np.abs(np.real(vals))
    cond = abs_imag_values > 2
    indexes = np.argwhere(cond).ravel()
    vals = np.delete(vals, indexes)
    vecs = np.delete(vecs, indexes, axis=1)

    # Separating modes into downwind and upwind
    eigval_angles = np.angle(vals)  # eigenvalue angles in the complex plane ([-π, π])
    eigval_angles = np.where(eigval_angles < 0, eigval_angles + 2 * np.pi, eigval_angles)
    tol = 0.0001
    cond = np.logical_and(eigval_angles > np.pi/4, eigval_angles < np.pi + tol)
    upwind_indexes = np.argwhere(cond).ravel()
    downwind_indexes = np.argwhere(np.logical_not(cond)).ravel()

    upwind_vals = vals[upwind_indexes]
    downwind_vals = vals[downwind_indexes]

    upwind_eigvecs = vecs[:, upwind_indexes]
    downwind_eigvecs = vecs[:, downwind_indexes]

    # Sorting the modes from less attenuated to more attenuated
    indexes = np.argsort(abs(np.imag(downwind_vals)), )
    downwind_vals = downwind_vals[indexes]
    downwind_eigvecs = downwind_eigvecs[:, indexes]

    indexes = np.argsort(abs(np.imag(upwind_vals)), )
    upwind_vals = upwind_vals[indexes]
    upwind_eigvecs = upwind_eigvecs[:, indexes]

    # Creating the Mode objects
    z_coords = np.array([node.z for node in mesh.nodes])
    left_modes = np.array([Mode(1, z_coords, complex(upwind_vals[i]),
                                upwind_eigvecs[:, i], complex(k0.value(0))) for i in range(0, len(upwind_vals))])
    right_modes = np.array([Mode(1, z_coords, complex(downwind_vals[i]),
                                 downwind_eigvecs[:, i], complex(k0.value(0))) for i in range(0, len(downwind_vals))])

    # Creating an AcousticField object
    acoustic_field = AcousticField(downwind_modes=right_modes, upwind_modes=left_modes)

    # Plotting the eigenvalues
    plt.figure()
    plt.scatter(np.real(upwind_vals), np.imag(upwind_vals), color='blue', label='upwind modes')
    plt.scatter(np.real(downwind_vals), np.imag(downwind_vals), color='red', label='downwind modes')
    plt.title("Mode eigenvalues λ")
    plt.xlabel("Real(λ)")
    plt.xlabel("Imag(λ)")
    plt.legend()
    plt.show()
    time5 = perf_counter()
    print('Separating the modes and creating classes took:', time5 - time4, 's')

    current, peak = tracemalloc.get_traced_memory()
    print(f"other: {current / 1024 ** 2:.2f} MB")
    print(f"other:    {peak / 1024 ** 2:.2f} MB")
    tracemalloc.reset_peak()

    return acoustic_field





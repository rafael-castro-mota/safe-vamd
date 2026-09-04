import numpy as np
from matplotlib import pyplot as plt
import tracemalloc

from safe_vamd.safe_computation import SAFEComputation

# Initializing a simulation with the Semi-Analytical Finite-Element method
safe_comp = SAFEComputation()

# Setting up the simulation with the text file containing the atmospheric background quantities and heights
input_file = "example_2_atmo_cond.txt"
safe_comp.setup(input_file, 180, rho_at_ground=1.235, g_value=9.81, pml_thick_factor=2)

# Assembling the matrices and solving the cubic eigenvalue problem (a ground impedance is used)
sigma_e = 841680
z_180hz = 0.218 * ((sigma_e / 180) ** (1 / 2)) * (1 + 1j)
acoustic_field = safe_comp.solve(ground_impedance_value=z_180hz, alpha_value=0.2, air_absorption=True)


# Saving the upwind and downwind modes in separate text files
safe_comp.save_modes_in_txt()

# Matching the downwind modes to a monopole at 3.6 m of height and amplitude of 1 kg/(m^3 s)
safe_comp.match_to_monopole(source_height=3.6, source_amplitude=1, side='downwind')


# Plotting a normalized acoustic pressure contour plot for the field matched to the monopole
rangos = np.linspace(0, 100, 1001)
x, z, p = acoustic_field.get_plot_data(rangos, side='downwind')


plot_limit = 0.1
ps = np.array(abs(p)/np.max(abs(p)))  # Normalizing the acoustic pressure w.r.t the maximum amplitude
ps = np.where(ps < plot_limit, ps, plot_limit)
levels = np.linspace(0.0, plot_limit, 100)
plt.contourf(x, z, ps, cmap='GnBu', levels=levels)
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.xlabel('Range [m]', fontsize=12)
plt.ylabel('Height [m]', fontsize=12)
plt.show()


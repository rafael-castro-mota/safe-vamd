import numpy as np
from matplotlib import pyplot as plt

from safe_vamd.safe_computation import SAFEComputation

# Initializing a VA mode computation with the Semi-Analytical Finite-Element method
safe_comp = SAFEComputation()

# Setting up the simulation with the .txt file containing the atmospheric background quantities and heights
input_file = "example_2_atmo_cond.txt"
safe_comp.setup(input_file, 180, rho_at_ground=1.235, g_value=9.81, pml_thick_factor=2)

# Assembling the matrices and solving the cubic eigenvalue problem (a ground impedance is used)
sigma_e = 841680
z_180hz = 0.218 * ((sigma_e / 180) ** (1 / 2)) * (1 + 1j)
acoustic_field = safe_comp.solve(ground_impedance_value=z_180hz, alpha_value=0.2, air_absorption=True)

# Saving the upwind and downwind modes in separate text files
safe_comp.save_modes_in_txt()
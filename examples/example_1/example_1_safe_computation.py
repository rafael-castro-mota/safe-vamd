import numpy as np
import time
from matplotlib import pyplot as plt
import tracemalloc

from safe_vamd.safe_computation import SAFEComputation

# Initializing a simulation with the Semi-Analytical Finite-Element method
safe_comp = SAFEComputation()

# Setting up the simulation with the text file containing the atmospheric background quantities and heights
input_file = 'example_1_atmo_cond.txt'
safe_comp.setup(input_file, 0.5, rho_at_ground=1.23, g_value=0, pml_thick_factor=2)

# Solving for the Vertical Atmospheric (VA) modes (hard ground)
acoustic_field = safe_comp.solve(alpha_value=0.4, air_absorption=False)

# Saving the upwind and downwind modes in separate text files
safe_comp.save_modes_in_txt()

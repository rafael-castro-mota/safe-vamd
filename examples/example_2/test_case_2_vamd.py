import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import tracemalloc

from safe_vamd.vamd import VAMD

# VAMD

# initializing VAMD and loading the modes
vamd = VAMD()

down_path = './downwind_modes.txt'
up_path = './upwind_modes.txt'
vamd.load_modes_from_txt(downwind_path=down_path, upwind_path=up_path)


# Mode filtering
vamd.mode_filter.plot_eigenvalues(upwind_color='gray', downwind_color='black')
time0 = perf_counter()
vamd.mode_filter.sort_by_imag_part(side='downwind')  # Sorting from less decaying to more decaying
vamd.mode_filter.filter_by_energy([0, 50], 0.85, side='downwind')  # removing PML modes
vamd.acoustic_field.downwind_modes = vamd.acoustic_field.downwind_modes[0:52]
time1 = perf_counter()
print('Mode filtering took:', time1 - time0, 's')
vamd.mode_filter.plot_eigenvalues(upwind_color='gray', downwind_color='red',
                                  downwind_label='downwind modes (filtered)', mk=1)


# Load samples from a .txt file
vamd.load_samples_from_txt('./samples/samples_180Hz.txt')

# Matching field to samples
acoustic_field = vamd.match_modes_to_samples(side='downwind', spread_3d=True)

# PLOTS

# Ploting a contour plot for the reconstructed acoustic pressure field
rangos = np.linspace(0, 100, 1001)
x, z, p = acoustic_field.get_plot_data(rangos, side='downwind')

plot_limit = 1
ps = np.array(abs(p)/np.max(abs(vamd.p_samples)))
ps = np.where(ps < plot_limit, ps, plot_limit)
levels = np.linspace(0.0, plot_limit, 100)

plt.contourf(x, z, ps, cmap='GnBu', levels=levels)
plt.ylim([0, 50])
plt.xlim([0, 100])
plt.xlabel("Range [m]", fontsize=12)
plt.ylabel("Height [m]", fontsize=12)
plt.show()

# Plotting VAMD results against experimental data
titles = ['2m horizontal', '10m horizontal', '30m horizontal']
horizontal_validation_scan_paths = ['./validation_samples/2m_horizontal_180Hz.txt',
                                    './validation_samples/10m_horizontal_180Hz.txt',
                                    './validation_samples/30m_horizontal_180Hz.txt']


for path, plot_title in zip(horizontal_validation_scan_paths, titles):
    data = np.loadtxt(path, skiprows=1, delimiter=';')
    ranges = data[:, 0].ravel()
    heights = data[:, 1].ravel()
    ps = data[:, 2].ravel() + 1j * data[:, 3].ravel()
    ps_ref = np.max(abs(ps))

    ps_vamd = np.array([acoustic_field.value_2d(x, z) for x, z in zip(ranges, heights)])
    ps_vamd_3d = np.array([acoustic_field.value_2d_with_3d_spread(x, z, 13) for x, z in zip(ranges, heights)])

    plt.figure()
    plt.title(plot_title)
    plt.scatter(ranges, 20 * np.log10(abs(ps)/ps_ref), color='black', label='Exp.', s=10)
    plt.plot(ranges, 20 * np.log10(abs(ps_vamd)/ps_ref), color='orange', linestyle='-.', label='VAMD w/o 3d spreading')
    plt.plot(ranges, 20 * np.log10(abs(ps_vamd_3d)/ps_ref), color='red', linestyle='--', label='VAMD w/ 3d spreading')
    plt.xlabel("Range [m]", fontsize=12)
    plt.ylabel("Transmission [dB]", fontsize=12)
    plt.xlim([5, 75])
    plt.ylim([-25, 5])
    plt.gca().set_box_aspect(0.229)
    plt.legend()

titles = ['30m vertical']
vertical_validation_scan_paths = ['./validation_samples/30m_vertical_180Hz.txt']

for path, plot_title in zip(vertical_validation_scan_paths, titles):
    plt.figure()
    data = np.loadtxt(path, skiprows=1, delimiter=';')
    ranges = data[:, 0].ravel()
    heights = data[:, 1].ravel()
    ps = data[:, 2].ravel() + 1j * data[:, 3].ravel()
    ps_ref = np.max(abs(ps))
    
    ps_vamd = np.array([acoustic_field.value_2d(x, z) for x, z in zip(ranges, heights)])
    ps_vamd_3d = np.array([acoustic_field.value_2d_with_3d_spread(x, z, 13) for x, z in zip(ranges, heights)])

    plt.scatter(20 * np.log10(abs(ps)/ps_ref), heights, color='black', label='Exp.')
    plt.plot(20 * np.log10(abs(ps_vamd)/ps_ref), heights, color='orange', linestyle='-.', label='VAMD w/o 3d spreading')
    plt.plot(20 * np.log10(abs(ps_vamd_3d)/ps_ref), heights, color='red', linestyle='--', label='VAMD w/ 3d spreading')
    plt.title(plot_title)
    plt.xlabel("Transmission [dB]", fontsize=12)
    plt.ylabel("Height [m]", fontsize=12)
    plt.legend()
plt.show()

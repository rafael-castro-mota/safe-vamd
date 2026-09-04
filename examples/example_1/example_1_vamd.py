import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import tracemalloc

from safe_vamd.vamd import VAMD
vamd = VAMD()

downwind_path = "downwind_modes.txt"
upwind_path = "upwind_modes.txt"
vamd.load_modes_from_txt(downwind_path=downwind_path, upwind_path=upwind_path)


vamd.mode_filter.plot_eigenvalues(upwind_color='grey', downwind_color='black')
time0 = perf_counter()
vamd.mode_filter.sort_by_imag_part(side="downwind")   # Sorting from less decaying to more decaying
vamd.mode_filter.filter_by_energy([0, 140 * 1000], 0.95, side="downwind")  # removing PML modes
#vamd.acoustic_field.downwind_modes = vamd.acoustic_field.downwind_modes[0:320]
vamd.acoustic_field.downwind_modes = vamd.acoustic_field.downwind_modes[0:380]

vamd.mode_filter.plot_eigenvalues(upwind_color='grey', downwind_color='red',
                                  downwind_label='downwind modes (filtered)', mk=True)

vamd.load_samples_from_txt("./samples/samples_x=7p5km+10km.txt")

acoustic_field = vamd.match_modes_to_samples(spread_3d=False)



rangos = np.linspace(0, 500 * 1000, 1001)
x, z, p = acoustic_field.get_plot_data(rangos, side='downwind')

# Normalizing pressure and plotting
plot_limit = 0.03
lee_reference = 0.04135302337341215
ps = np.array(abs(p)/lee_reference)
ps = np.where(ps < plot_limit, ps, plot_limit)
levels = np.linspace(0.0, plot_limit, 1000)
plt.contourf(x, z, ps, cmap='GnBu_r', levels=levels)
plt.ylim([0, 140 * 1000])
plt.xlim([0, 500 * 1000])
plt.show()


validation_paths = ['./validation_samples/height=1km.txt', './validation_samples/height=100km.txt']
heights = [1000, 100 * 1000]


for path, sl_hs in zip(validation_paths, heights):
    data = np.loadtxt(path, skiprows=1, delimiter=';')
    #ranges = data[:, 0].ravel()
    #heights = data[:, 1].ravel()
    ps = data[:, 2].ravel() + 1j * data[:, 3].ravel()

    ranges = np.linspace(0, 1000 * 1000, 2000)
    heights = sl_hs*np.ones(len(ranges))

    ps_vamd = np.array([acoustic_field.value_2d(x, z) for x, z in zip(ranges, heights)]).ravel()

    p_ref = np.max(abs(ps))

    plt.figure()
    #plt.plot(ranges, 20 * np.log10(abs(ps) / p_ref), color='black')
    plt.plot(ranges, 20 * np.log10(abs(ps_vamd) / p_ref), color='red', linestyle="--")
    plt.xlim([0, 1000 * 1000])
    plt.ylim([-60, 10])
    plt.gca().set_box_aspect(0.229)
plt.show()

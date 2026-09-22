Example 2 - Experimental outdoor reconstruction
===============================================
The work in http://dx.doi.org/10.2139/ssrn.7427778 employed the VAMD method to reconstruct
the acoustic field induced by an outdoor source from experimental data.

This example follows the workflow depicted in Schematic 1 in (inserir link) and
the results are detailed in Section 3.2 of the same article.

Computing the Vertical Atmospheric (VA) Modes
---------------------------------------------

A Safe computation must be initiated.

.. code-block:: python

    from safe_vamd.safe_computation import SAFEComputation
    safe_comp = SAFEComputation()

The setup of the computation can be run with:

.. code-block:: python

    input_file = "example_2_atmo_cond.txt"
    safe_comp.setup(input_file, 180, rho_at_ground=1.235, g_value=9.81, pml_thick_factor=2)

, where the excitation frequency has been set to 180 :math:`Hz`. The parameters *rho_at_ground* and *g_value* are the
the air density at ground level and the acceleration due to gravity, respectively, and were set to to 1.235
:math:`kgm^{-3}` and 9.81 :math:`ms^{-2}`.
The thickness of the Perfectly Matched Layer (PML) has been set to two times the greatest wavelength
(*pml_thickness_factor=2*).
The parameter *input_file* is a path to a *.txt* file with the background atmospheric conditions,
given in the example folder under the name
"example_2_atmo_cond.txt". It has the following format:

.. image:: figures/example_2_atmo_cond.png

After setting up, the VA modes can be computed with:

.. code-block:: python

    sigma_e = 841680
    z_180hz = 0.218 * ((sigma_e / 180) ** (1 / 2)) * (1 + 1j)
    acoustic_field = safe_comp.solve(ground_impedance_value=z_180hz, alpha_value=0.2, air_absorption=True)

, where the parameter *alpha_value* is a term for added absorption in the Perfectly Matched Layer that was set to 0.2
:math:`dB/wavelength`.
In this example, air absorption is accounted for and a ground impedance boundary condition was set, with the
parameter *ground_impedance_value* asigned a value computed
with Eq. 14 in https://doi.org/10.1016/0022-460X(85)90538-3 and :math:`\sigma_e=841680 Pasm^{−2}`.

After the modes have been computed, a figure with the mode eignevalues plotted in the complex-plane will show up.
The user can double check if the modes have been correctly separated into the downwind and upwind mode basis.

.. image:: figures/example_2_safecomp_eigenvalues.png

The VA modes can be exported into *.txt* files without prior weighing:

.. code-block:: python

    safe_comp.save_modes_in_txt()

Matching the VA modes to the acoustic pressure samples.
-------------------------------------------------------

Vertical Atmospheric Mode Decomposition can be applied with the methods in the *VAMD* class:

.. code-block:: python

    from safe_vamd.vamd import VAMD
    vamd = VAMD()


After the VA modes have been computed and stored in *.txt* files, they can be loaded:

.. code-block:: python

    downwind_path = "downwind_modes.txt"
    upwind_path = "upwind_modes.txt"
    vamd.load_modes_from_txt(downwind_path=down_path, upwind_path=up_path)

and then filtered by resorting to the methods in the *mode_filter* attribute:

.. code-block:: python

    n_modes = 52
    vamd.mode_filter.sort_by_imag_part(side='downwind')  # Sorting from less decaying to more decaying
    vamd.mode_filter.filter_by_energy([0, 50], 0.85, side='downwind')  # removing PML modes
    vamd.acoustic_field.downwind_modes = vamd.acoustic_field.downwind_modes[0:n_modes] # truncating mode basis

To ensure the correct filtering of the modes, their eigenvalues can be plotted before and after
filtering has been applied:

.. code-block:: python

    vamd.mode_filter.plot_eigenvalues(upwind_color='grey', downwind_color='black'
    ### any mode fitlering here ###
    vamd.mode_filter.plot_eigenvalues(upwind_color='grey', downwind_color='red',
                                  downwind_label='downwind modes (filtered)', mk=True)

, which for this example would generate the following image:

.. image:: figures/example_1_eigenvalues.png

The samples can be loaded with:

.. code-block:: python

    vamd.load_samples_from_txt("./samples/samples_180Hz.txt")

, where *"./samples/samples_180Hz.txt"* is a path to a *.txt* file in the example folder with
the acoustic transfer functions. It has the following format:

.. image:: figures/example_2_samples.png

The samples can be matched to the samples with:

.. code-block:: python

    acoustic_field = vamd.match_modes_to_samples(spread_3d=True)

, where the parameter *spread_3d* was set to *True* to account for
spherical spreading in the mode decomposition procedure.

After the VA modes have been matched to the samples, the example script also includes post-processing and plotting.
The generated contour plot for the normalized acoustic pressure is presented here:

.. image:: figures/example_2_matched_to_samples.png

, along with the transmission (in :math:`dB`) for the horizontal line 2 m from the ground:

.. image:: figures/transmisison_example_2_2m_h.png

,where the black dots represent the transmission computed with experimental data,
and the red and orange dashed lines represent the fields reconstructed via mode decomposition,
with and without a spherical spreading correction factor, respectively.


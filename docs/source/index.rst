.. AtmoSAFE documentation master file, created by
   sphinx-quickstart on Thu Aug 13 12:12:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SAFE-VAMD
====================================================

This python package computes acoustic vertical modes propagating on a stratified, range-indepedent atmosphere and
can, subsequently, project acoustic pressure data onto them, resulting in a reconstructed field. The data can either
be analytical, numerical or experimental.

**The package features:**

    - The ability to compute vertical atmospheric (VA) mode basis with the Semi-Analytical Finite-Element (SAFE) method.
    - Acoustic field reconstruction by decomposition of acoustic pressure data, either numerical or experimental, onto VA mode basis.



Installation
------------

The module can be easily installed using PiPy:

.. code-block:: console

   >> pip install safe-vamd

The source code is available at `GitHub <https://github.com/rafael-castro-mota/safe-vamd>`_\.

Theory
------------

The Semi-Analytical Finite-Element (SAFE) method was developed in https://doi.org/10.1121/10.0002912 and
extended to use a ground impedance in https://doi.org/10.1121/10.0003567
It is also summarized in https://doi.org/http://dx.doi.org/10.2139/ssrn.7427778.

Acoustic field reconstruction through Vertical Atmospheric Mode Decomposition (VAMD) has been demonstrated
in https://doi.org/10.1016/j.enganabound.2025.106308, for numerical data, and in
https://doi.org/http://dx.doi.org/10.2139/ssrn.7427778, for experimental data.

Acknowledgement
---------------

The research that led to this package was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under project numbers 536842818 ('Acoustic modal expansion for low-frequency sound in ducted atmospheres') and 541019206 ('UAV-based near-field to far-field transformation for a detailed characterization of large outdoor emitters in operation').

Examples
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   example_1
   example_2

The Module
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules



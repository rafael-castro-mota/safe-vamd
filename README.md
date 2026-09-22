SAFE-VAMD
====================================================

This python package computes acoustic vertical modes propagating on a stratified, range-independent atmosphere and
can, subsequently, project acoustic pressure data onto them, resulting in a reconstructed far-field. The data can either
be analytical, numerical or experimental.

**The package features:**

- The ability to compute vertical atmospheric (VA) mode basis with the
Semi-Analytical Finite-Element (SAFE) method.
- Acoustic field reconstruction by decomposition of acoustic pressure data,
either numerical or experimental, onto VA mode basis.



Installation
------------

The module can be easily installed using PiPy:
>> pip install safe-vamd

The documentation is available at [ReadTheDocs](https://safe-vamd.readthedocs.io/en/latest/).

Theory
------------

The Semi-Analytical Finite-Element (SAFE) method was developed in https://doi.org/10.1121/10.0002912 and
extended to use a ground impedance in https://doi.org/10.1121/10.0003567.
It is also summarized in https://doi.org/http://dx.doi.org/10.2139/ssrn.7427778.

Acoustic field reconstruction through Vertical Atmospheric Mode Decomposition (VAMD) has been demonstrated
in https://doi.org/10.1016/j.enganabound.2025.106308, for numerical data, and in
https://doi.org/http://dx.doi.org/10.2139/ssrn.7427778, for experimental data.

Acknowledgement
---------------

The research that led to this package was funded by the Deutsche Forschungsgemeinschaft
(DFG, German Research Foundation) under project numbers 536842818
('Acoustic modal expansion for low-frequency sound in ducted atmospheres')
and 541019206 ('UAV-based near-field to far-field transformation
for a detailed characterization of large outdoor emitters in operation').

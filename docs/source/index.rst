.. AtmoSAFE documentation master file, created by
   sphinx-quickstart on Thu Aug 13 12:12:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SAFE-VAMD
====================================================

This python package computes acoustic vertical modes propagating on a stratified, range-indepedent atmosphere and
can, subsequently, project acoustic pressure data onto them, resulting in a reconstructed field. The data can either
be numerical or experimental.

**The package features:**

    - The ability to compute vertical atmopsheric (VA) mode basis with the Semi-Analytical Finite-Element (SAFE) method.
    - Decompose acoustic pressure data, either numerical or experimental, onto VA mode basis with the Vertical Atmospheric Mode Decomposition (VAMD) method.



Installation
------------

The module can be easily installed using PiPy:

.. code-block:: console

   >> pip install safe-vamd

The source code is available at `GitHub <https://github.com/rafael-castro-mota/safe-vamd>`_\.

Acknowledgement
---------------

The research that led to this package was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under project numbers 536842818 ('Acoustic modal expansion for low-frequency sound in ducted atmospheres') and 541019206 ('UAV-based near-field to far-field transformation for a detailed characterization of large outdoor emitters in operation').

Examples
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   example_1

The Module
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules



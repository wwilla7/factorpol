Factor-Pol
==============================

.. image:: https://readthedocs.org/projects/factorpol/badge/?version=latest
    :target: https://factorpol.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/wwilla7/factorpol/actions/workflows/CI.yaml/badge.svg
    :target: https://github.com/wwilla7/factorpol/actions/workflows/CI.yaml

.. image:: https://codecov.io/gh/wwilla7/factorpol/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/wwilla7/factorpol/branch/main
    :alt: codecov

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7750889.svg
   :target: https://doi.org/10.5281/zenodo.7750889

A Fast Atom-Centered Typed Isotropic Ready-to-use Polarizable Electrostatic Model. Factor-Pol Model

The latest implementation lives at `dpolfit <https://github.com/wwilla7/dpolfit>`_.

The model is described at: 

**A Fast, Convenient, Polarizable Electrostatic Model for Molecular Dynamics**

    Liangyue Wang, Michael Schauperl, David L. Mobley, Christopher Bayly, and Michael K. Gilson

    Journal of Chemical Theory and Computation 2024 20 (3), 1293-1305 `DOI: 10.1021/acs.jctc.3c01171 <https://pubs.acs.org/doi/10.1021/acs.jctc.3c01171>`_.

We present an efficient polarizable electrostatic model using direct polarization[1], OpenFF SMIRNOFF[2] typed polarizabilities, and a new AM1-BCC[3,4]-style charge model for improved electrostatics in molecular dynamics (MD) simulations.

This toolkit and `documentation <https://factorpol.readthedocs.io/en/latest>`_ are currently under continuous development and improvement.

Some examples can be found here: `example <examples>`_.

A post presentation can be found here: `poster <https://zenodo.org/record/7750889>`_.




References
----------

| [1] Straatsma, T.; McCammon, J. Molecular Simulation 1990, 5, 181–192.
| [2] Mobley, D. L.; Bannan, C. C.; Rizzi, A.; Bayly, C. I.; Chodera, J. D.; Lim, V. T.; Lim, N. M.; Beauchamp, K. A.; Slochower, D. R.; Shirts, M. R., et al. Journal of chemical theory and computation 2018, 14, 6076–6092.
| [3] Jakalian, A.; Bush, B. L.; Jack, D. B.; Bayly, C. I. Journal of computational chemistry 2000, 21, 132–146.
| [4] Jakalian, A.; Jack, D. B.; Bayly, C. I. Journal of computational chemistry 2002, 23, 1623–1641.

Copyright
---------

Copyright (c) 2023, Liangyue (Willa) Wang


Acknowledgements
----------------

Project based on the
`Computational Molecular Science Python Cookiecutter <https://github.com/molssi/cookiecutter-cms>`_ version 1.1.

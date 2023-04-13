Examples
=========

This directory contains six examples to use `factorpol` package.

Notebooks
-----------------

1. `00-generate-qm-reference.ipynb <00-generate-qm-reference.ipynb>`_

   This example is used to generate QM reference data for trainingset. The QM reference data is used to calculate the difference between QM and MM energies. The difference is used to optimize polarizability parameters.

2. `01-derive-polarizability.ipynb <01-derive-polarizability.ipynb>`_

   This example is used to optimize polarizability parameters for a trainingset using "Nelder-Mead" optimizer.

3. `02-RESP-dPol.ipynb <02-RESP-dPol.ipynb>`_

   This notebook provides an example to generate RESP-dPol partial charges.

4. `03-AM1-BCC-dPol.ipynb <03-AM1-BCC-dPol.ipynb>`_

   This notebook provides an example to generate AM1-BCC-dPol partial charges.

5. `04-parameterize-for-simulation.ipynb <04-parameterize-for-simulation.ipynb>`_

   This example is used to generate force field file for OpenMM and MPID plugin

6. `05-optimize-pols-vdiff.ipynb <05-optimize-pols-vdiff.ipynb>`_

   This example is used to optimize polarizability parameters for a trainingset without using "Nelder-Mead" optimizer.


Relevant data
-----------------
| Polarizability: `ret_alphas.csv <ret_alphas.csv>`_
| Bond charge correction with polarizability: `ret_bccs.csv <ret_bccs.csv>`_
| Force field `.xml` for OpenMM and `MPID plugin <https://github.com/andysim/MPIDOpenMMPlugin>`_: `forcefield.xml <forcefield.xml>`_

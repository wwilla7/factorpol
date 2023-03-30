"""
A module to generate partial charges to use with typed polarizabilities.
"""


import copy
from typing import List, Tuple

import numpy as np
import pint
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from scipy.spatial import distance

from factorpol.utilities import (canonical_ranking, coulomb_scaling,
                                 flatten_a_list, pair_equivalent,
                                 Polarizability, smirnoff_labels)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


class ChargeTrainer:
    """
    A Class contains all information needed to do ESP-fitting related calculations.

    **All operations use atomic unit**

    Parameters
    ----------
    record: MoleculeESPRecord
        MoleculeESPRecord contains all EPS-fitting related reference data.

    polarizability: Polarizability
        Polarizabilities for all polarizability related operation

    off_forcefield: ForceField
        An OpenFF ForceField used for labeling SMIRNOFF patterns.

    coulomb14scale: float
        A scaling factor to scale 1-4 coulomb interactions.
        Default is 0.5
        Commonly used values include 0.83333

    """
    def __init__(
            self,
            record: MoleculeESPRecord,
        polarizability: Polarizability,
        off_forcefield: ForceField,
        coulomb14scale: float = 0.5,
    ):

        self.record = record
        self.tagged_smiles = self.record.tagged_smiles
        self.atomcrds = Q_(self.record.conformer, "angstrom").to("bohr").magnitude
        self.gridcrds = (
            Q_(self.record.grid_coordinates, "angstrom").to("bohr").magnitude
        )
        self.npoints = len(self.gridcrds)
        self.natoms = len(self.atomcrds)
        self.esp_values = Q_(self.record.esp, "e/a0").magnitude.reshape(-1)
        self._r_ij = distance.cdist(self.atomcrds, self.gridcrds)
        self.grid_to_atom = np.reciprocal(self._r_ij.T)
        self.r_ij = np.array(
            [
                [self.atomcrds[j] - self.gridcrds[i] for i in range(self.npoints)]
                for j in range(self.natoms)
            ]
        )
        self.rec_rij = np.reciprocal(self._r_ij)
        self.r_ij3 = np.power(self._r_ij, -3).reshape([self.natoms, self.npoints, 1])
        self.r_jk = np.array(
            [
                [self.atomcrds[j] - self.atomcrds[k] for k in range(self.natoms)]
                for j in range(self.natoms)
            ]
        )  # distance vector
        self._r_jk = distance.cdist(self.atomcrds, self.atomcrds)  # distance values
        self.r_jk3 = np.where(
            self._r_jk == 0.0, self._r_jk, np.power(self._r_jk, -3)
        ).reshape(
            [self.natoms, self.natoms, 1]
        )  # r_{jk}^3

        # reconstruct an OpenFF molecule and put in correct coordinates.
        offmol = Molecule.from_mapped_smiles(self.tagged_smiles)
        offmol.generate_conformers(n_conformers=1)
        offmol.conformers.clear()
        offmol.conformers.append(self.atomcrds * unit.bohr)
        self.offmol = copy.deepcopy(offmol)

        self.rdmol = self.offmol.to_rdkit()
        self.qcmol = self.offmol.to_qcschema()
        self.symbols = self.qcmol.symbols
        self.equivalent_atoms = pair_equivalent(canonical_ranking(self.rdmol))
        self.n_equivalent = len(self.equivalent_atoms)
        self.net_charge = self.qcmol.molecular_charge
        self.smirnoff_patterns = smirnoff_labels(self.offmol, off_forcefield)
        self.coulomb14scale = coulomb14scale
        self.coulomb_scaling_matrix = None
        self.alphas = [
            polarizability.parameters[p].to("bohr**3").magnitude
            for p in self.smirnoff_patterns
        ]

    @property
    def smiles(self) -> str:
        """

        Returns
        -------
        str
            SMILES string without explicit hydrogens.

        """
        return self.offmol.to_smiles(explicit_hydrogens=False)

    @property
    def polar_region(self) -> List:
        """
        A method used to select polar region for the second stage RESP-dPol fit.

        Returns
        -------
        List
            Returns a list of atoms that are defined as in polar region.

        """
        forced_symmtry = set(flatten_a_list(self.equivalent_atoms))
        ret = list(set(range(self.offmol.n_atoms)) - forced_symmtry)
        return ret

    @property
    def resp_dpol(self) -> pint.Quantity:
        """
        A method to generate RESP-dPol partial charges by fitting to baseline QM ESPs

        Returns
        -------
        pint.Quantity
            Returns RESP-dPol partial charges

        """
        _, _, preq = self.plain_esp_charges()
        _, _, qd = self.derive_resp_dpol(pre_charge=preq)
        return Q_(qd, ureg.elementary_charge)

    @property
    def respdpol_dipoles(self) -> pint.Quantity:
        """

        Returns
        -------
        pint.Quantity
            Molecular dipole moment calculated using RESP-dPol charges.

        """
        ret = self.calc_molecular_dipoles(self.resp_dpol)
        return ret

    @property
    def mm_base_esps(self) -> pint.Quantity:
        """

        Returns
        -------
        pint.Quantity
            Calculated MM ESPs without polarizability

        """
        ret = self.calc_Esps(self.resp_dpol.magnitude)
        return Q_(ret, "e*a0")

    @property
    def mm_dpol_esps(self) -> pint.Quantity:
        """

        Returns
        -------
        pint.Quantity
            Calculated MM ESPs with polarizability

        """
        ret, _ = self.calc_Esps_dpol(self.resp_dpol.magnitude)
        return Q_(ret, "e*a0")

    def plain_esp_charges(self) -> Tuple:
        """
        Derive unconstrained ESP-fitting charges from baseline QM ESPs.

        Returns
        -------
        ndarray, ndarray, ndarray
            matrix, vector, solution

        """
        dimension = self.natoms + 1
        matrix = np.zeros([dimension, dimension])
        tmp = np.einsum("jk,jm->km", self.grid_to_atom, self.grid_to_atom)

        matrix[: self.natoms, : self.natoms] = tmp

        # Lagrange
        matrix[self.natoms, :] = 1.0
        matrix[:, self.natoms] = 1.0
        matrix[self.natoms, self.natoms] = 0.0

        vector = np.zeros(dimension)

        vector[: self.natoms] = np.einsum("ik,i->k", self.grid_to_atom, self.esp_values)
        vector[self.natoms] = float(self.net_charge)

        esp_charge = np.linalg.solve(matrix, vector)[: self.natoms]

        return matrix, vector, esp_charge

    def forced_symmetry_esp_charges(self) -> Tuple:
        """
        Derived ESP-fitting partial charges from baseline QM ESPs with forced symmetry

        Returns
        -------
        ndarray, ndarray, ndarray
            matrix, vector, solution

        """
        dimension = self.natoms + 1 + self.n_equivalent
        matrix = np.zeros([dimension, dimension])
        tmp = np.einsum("jk,jm->km", self.grid_to_atom, self.grid_to_atom)

        matrix[: self.natoms, : self.natoms] = tmp

        # Lagrange
        matrix[self.natoms, :] = 1.0
        matrix[:, self.natoms] = 1.0
        matrix[self.natoms, self.natoms] = 0.0

        # chemical equivalent atoms
        if self.n_equivalent == 0:
            pass
        else:
            for idx, pair in enumerate(self.equivalent_atoms):
                matrix[self.natoms + 1 + idx, :] = 0.0
                matrix[:, self.natoms + 1 + idx] = 0.0

                matrix[self.natoms + 1 + idx, pair[0]] = 1.0
                matrix[self.natoms + 1 + idx, pair[1]] = -1.0
                matrix[pair[0], self.natoms + 1 + idx] = 1.0
                matrix[pair[1], self.natoms + 1 + idx] = -1.0

        vector = np.zeros(dimension)

        vector[: self.natoms] = np.einsum("ik,i->k", self.grid_to_atom, self.esp_values)
        vector[self.natoms] = float(self.net_charge)

        esp_charge = np.linalg.solve(matrix, vector)[: self.natoms]

        return matrix, vector, esp_charge

    def derive_resp_dpol(self, pre_charge: np.ndarray) -> Tuple:
        r"""
        Derive RESP-dPol partial charges from baseline QM ESPs.

        .. math:: \chi^2 = \sum_{i=1}^{m} (V_{\mathrm{QM, i}} - V_{\mathrm{perm, i}} - V_{\mathrm{ind, i}}) + \lambda({\sum}_{j=1}^{n}q_j - q_{\mathrm{tot}}) + a\sum_{j=1}^{n}(\sqrt{q_j^2 + b^2} - b)


        first stage: a = 0.005 a.u., b = 0.1 a.u.

        second stage: a = 0.01 a.u., b = 0.1 a.u.

        Parameters
        ----------
        pre_charge: ndarray
            Plain ESP charges as a starting point

        Returns
        -------
        ndarray, ndarray, ndarray
            matrix, vector, solution

        """

        _, intra_e = self.calc_Esps_dpol(pre_charge)

        dimension = self.natoms + 1 + self.n_equivalent + len(self.polar_region)
        matrix = np.zeros([dimension, dimension])
        tmp = np.einsum("jk,jm->km", self.grid_to_atom, self.grid_to_atom)

        matrix[: self.natoms, : self.natoms] = tmp

        # Lagrange
        matrix[self.natoms, :] = 1.0
        matrix[:, self.natoms] = 1.0
        matrix[self.natoms, self.natoms] = 0.0

        # chemical equivalent atoms
        if self.n_equivalent == 0:
            pass
        else:
            for idx, pair in enumerate(self.equivalent_atoms):
                matrix[self.natoms + 1 + idx, :] = 0.0
                matrix[:, self.natoms + 1 + idx] = 0.0

                matrix[self.natoms + 1 + idx, pair[0]] = 1.0
                matrix[self.natoms + 1 + idx, pair[1]] = -1.0
                matrix[pair[0], self.natoms + 1 + idx] = 1.0
                matrix[pair[1], self.natoms + 1 + idx] = -1.0

        vector = np.zeros(dimension)
        right_part = self.esp_values - intra_e
        vector[: self.natoms] = np.einsum("ik,i->k", self.grid_to_atom, right_part)

        vector[self.natoms] = float(self.net_charge)

        # constrain polar region charges
        charge_to_be_fixed = pre_charge[self.polar_region]

        for idx, pol_idx in enumerate(self.polar_region):
            matrix[dimension - len(self.polar_region) + idx, pol_idx] = 1.0
            matrix[pol_idx, dimension - len(self.polar_region) + idx] = 1.0
            vector[dimension - len(self.polar_region) + idx] = charge_to_be_fixed[idx]

        esp_charge = np.linalg.solve(matrix, vector)[: self.natoms]

        return matrix, vector, esp_charge

    def calc_Esps_dpol(self, partial_charge: np.ndarray) -> Tuple:
        """
        Calculate Coulomb potentials on grid points using polarizabilities and input partial charges.

        Parameters
        ----------
        partial_charge: np.ndarray
            Input partial charges to compute ESPs on grid points.

        Returns
        -------
        ndarray, ndarray
            Total EPSs, ESPs from induced dopoles.

        """

        self.coulomb_scaling_matrix = coulomb_scaling(
            self.rdmol, coulomb14scale=self.coulomb14scale
        )
        # # Initialized electric field matrix
        efield = self._calc_efield(partial_charge)

        # Base Esps by fixed point charges
        base_esps = self.calc_Esps(partial_charge)

        # Esps by intramolecular polarization
        dpol_esps = np.zeros(self.npoints)

        alphas = copy.deepcopy(self.alphas)

        for j in range(self.natoms):
            for i in range(self.npoints):
                dpol_esps[i] += (
                    alphas[j] * np.dot(efield[j], self.r_ij[j, i]) * self.r_ij3[j, i]
                )

        ret = base_esps + dpol_esps
        return ret, dpol_esps

    def _calc_efield(self, partial_charge: np.ndarray) -> np.ndarray:
        """
        A method to compute electric field generated by permanent electrostatics.

        Parameters
        ----------
        partial_charge: ndarray
            Input partial charges to generate local electric fields.

        Returns
        -------
        ndarray
            Returns local electric fields generated by permanent partial charges.

        """

        efield = np.zeros((self.natoms, 3))
        for k in range(self.natoms):
            for j in range(self.natoms):
                efield[k] += (
                    partial_charge[j]
                    * self.r_jk[k, j]
                    * self.r_jk3[k, j]
                    * self.coulomb_scaling_matrix[k, j]
                )
        return efield

    def calc_Esps(self, partial_charge: np.ndarray) -> np.ndarray:
        r"""

        A method to compute Coulomb potentials generated by permanent partial charges

        .. math:: V_{i} = \sum_{j=1}^{n} \frac{q_{j}}{r_{ij}}

        Parameters
        ----------
        partial_charge: ndarray
            Input partial charges to generate ESPs on grid points.

        Returns
        -------
        ndarray
            Returns computed ESPs without polarizability.

        """

        esps = np.dot(partial_charge, self.rec_rij)

        return esps

    def calc_molecular_dipoles(self, partial_charges: np.ndarray) -> pint.Quantity:
        """
        Compute molecular dipole moment

        Parameters
        ----------
        partial_charges: ndarray
            Input partial charges to calculate molecular dipole moments

        Returns
        -------
        pint.Quantity
            Returns molecular dipole moment.

        """

        if isinstance(partial_charges, pint.Quantity):
            qs = partial_charges.magnitude
        ret = Q_(
            np.linalg.norm(
                np.sum(
                    np.multiply(qs.reshape(-1, 1), self.atomcrds),
                    axis=0,
                )
            ),
            "e*a0",
        ).to("debye")
        return ret

    def calc_molecular_dipoles_dpol(self, partial_charges: np.ndarray) -> pint.Quantity:
        r"""
        Compute molecular dipole moments with polarizability and permanent partial charges

        .. math:: \mu = \sum{j=1}^{n}(q_j + \mu_\mathrm{ind, j})~\mathrm{r}_j

        Parameters
        ----------
        partial_charges: ndarray
            Input partial charges to calculate molecular dipole moments

        Returns
        -------
        pint.Quantity
            Returns molecular dipole moment.

        """

        if isinstance(partial_charges, pint.Quantity):
            qs = partial_charges.magnitude

        efield = self._calc_efield(partial_charges)
        alphas = copy.deepcopy(self.alphas)
        alphas = np.array(alphas).reshape(-1)
        induced_dipoles = np.linalg.norm(np.multiply(efield, alphas), axis=-1).reshape(
            -1
        )
        aqs = qs.reshape(-1) + induced_dipoles
        ret = Q_(
            np.linalg.norm(
                np.sum(
                    np.multiply(aqs, self.atomcrds),
                    axis=0,
                )
            ),
            "e*a0",
        ).to("debye")
        return ret

    def calc_Esps_mpol(self, partial_charge: np.ndarray) -> Tuple:
        r"""
        Calculate Coulomb potentials using mutual polarization

        .. math::
                  {\mathbf{\mu}_{ind,j}} = {\alpha_j} {\mathbf{E}_j}

        Electric field produced by induced dipole moments:

        .. math::
                  \mathbf{E}_{\mu} = \frac{1}{\mathbf{r}^3}[(3\mathbf{\mu}\cdot r) r - \mathbf{\mu}]

        Parameters
        ----------
        partial_charge: ndarray
            Input partial charges to compute MM ESPs on grid points

        Returns
        -------
        ndarray, ndarray
            Total ESPs and ESPs generated by induced dipoles

        """

        alphas = copy.deepcopy(self.alphas)
        alphas = np.array(alphas).reshape(-1, 1)
        E = self._calc_efield(partial_charge)
        previous_induced = np.multiply(alphas, E)

        converge = False
        criteria = 1e-7
        step = 0

        while converge == False:
            step += 1
            induced_field = self._calc_efield_by_dipole(previous_induced)
            this_induced_dipole = np.multiply(alphas, (E + induced_field))

            this_induced_dipole_norm = np.linalg.norm(this_induced_dipole, axis=-1)
            last_induced_dipole_norm = np.linalg.norm(previous_induced, axis=-1)

            if np.allclose(
                this_induced_dipole_norm, last_induced_dipole_norm, atol=criteria
            ):
                converge = True
            else:
                previous_induced = this_induced_dipole.copy()

            if step > 50:
                converge = True

            efield = E + induced_field

            charge_esp = self.calc_Esps(partial_charge)
            dipole_esp = np.zeros(self.npoints)

            for j in range(self.natoms):
                for i in range(self.npoints):
                    dipole_esp[i] += (
                        alphas[j]
                        * np.dot(efield[j], self.r_ij[j, i])
                        * self.r_ij3[j, i]
                    )

            ret = charge_esp + dipole_esp
            return ret, dipole_esp

    def _calc_efield_by_dipole(self, dipole: np.ndarray) -> np.ndarray:
        """
        Calculate electric fields generated by induced dipoles with direct polarization

        Parameters
        ----------
        dipole: ndarray
            Input atomic dipole to compute electric fields from.

        Returns
        -------
        ndarray
            Returns computed electric field tensor.

        """

        induced_field = np.zeros((self.natoms, 3))

        for k in range(self.natoms):
            for j in range(self.natoms):
                induced_field[k] += (
                    self.r_jk3[k, j]
                    * (dipole[j] * self.r_jk[k, j] * self.r_jk[k, j] - dipole[j])
                    * self.coulomb_scaling_matrix[k, j]
                )
        return induced_field

"""
A module to generate partial charges to use with typed polarizabilities.
"""


import copy
import os
from enum import Enum
from typing import List

import numpy as np
import pint
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit import Molecule, ForceField
from openff.units import unit
from scipy.spatial import distance

from factorpol.utilities import (
    canonical_ranking,
    coulomb_scaling,
    flatten_a_list,
    pair_equivalent,
    PolarizabilityType,
    smirnoff_labels,
)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity



class ChargeTrainer:
    def __init__(
        self,
        record: MoleculeESPRecord,
        polarizability_type: Enum,
        offmol: Molecule,
        ff: ForceField,
    ):
        """
        A class to generate QM ESPs derived partial charges in the context of direct polarization.
        
        Notes: this class use atomic unit for all calculations.
        
        :param record: `openff.recharge.esp.storage.MoleculeESPRecord` that contains all QM reference data 
                        for generating QM ESPs derived partial charges
        :param polarizability_type: The polarizability typing scheme of choice.
        """
        self.record = record
        self.polarizability_type = polarizability_type
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

        # tmp_mol = Molecule.from_smiles(self.tagged_smiles)
        # smiles = tmp_mol.to_smiles(explicit_hydrogens=False)
        # offmol = Molecule.from_smiles(smiles)
        if offmol.conformers == None:
            offmol.generate_conformers(n_conformers=1)
            offmol.conformers.clear()
            offmol.conformers.append(self.atomcrds * unit.angstrom)
        else:
            pass
        self.offmol = copy.deepcopy(offmol)
        self.rdmol = self.offmol.to_rdkit()
        self.qcmol = self.offmol.to_qcschema()
        self.symbols = self.qcmol.symbols
        self.equivalent_atoms = pair_equivalent(canonical_ranking(self.rdmol))
        self.n_equivalent = len(self.equivalent_atoms)
        self.net_charge = self.qcmol.molecular_charge
        # self.parameters_path = "default"
        self.coulomb_scaling_matrix = None
        self.smirnoff_patterns = smirnoff_labels(self.offmol, ff)

    @property
    def coulomb14scale(self):
        return self._coulomb14scale

    @coulomb14scale.setter
    def coulomb14scale(self, value):
        self._coulomb14scale = value

    @property
    def smiles(self) -> str:
        return self.offmol.to_smiles(explicit_hydrogens=False)

    @property
    def polar_region(self) -> List:
        forced_symmtry = set(flatten_a_list(self.equivalent_atoms))
        ret = list(set(range(self.offmol.n_atoms)) - forced_symmtry)
        return ret

    @property
    def partial_charges_espfit(self) -> pint.Quantity:
        _, _, preq = self.free_esp_charges()
        _, _, qd = self.resp_style_dpolq(pre_charge=preq)
        return Q_(qd, ureg.elementary_charge)

    @property
    def alphas(self) -> List:
        return self._alphas

    @alphas.setter
    def alphas(self, value: Enum):
        if self.polarizability_type == PolarizabilityType.Element:
            self._alphas = [
                value.parameters[ele].to("bohr**3").magnitude for ele in self.symbols
            ]
        elif self.polarizability_type == PolarizabilityType.SMIRNOFF:
            self._alphas = [
                value.parameters[sf].to("bohr**3").magnitude
                for sf in self.smirnoff_patterns
            ]
        else:
            raise NotImplementedError

    @property
    def molecular_dipole_esp(self) -> pint.Quantity:
        ret = self.calc_molecular_dipoles(self.partial_charges_espfit)
        return ret

    @property
    def molecular_dipole_bcc(self) -> pint.Quantity:
        ret = self.calc_molecular_dipoles(self.partial_charges_bcc)
        return ret

    @property
    def mm_base_esps(self) -> pint.Quantity:
        ret = self.calc_Esps(self.partial_charges_espfit.magnitude)
        return Q_(ret, "e*a0")

    @property
    def mm_dpol_esps(self):
        ret, _ = self.calc_Esps_dpol(self.partial_charges_espfit.magnitude)
        return Q_(ret, "e*a0")

    @property
    def partial_charges_bcc(self):
        """
        Generate AM1-BCC-dPol partial charges
        :return: AM1-BCC-dPol partial charges
        """
        from factorpol.bcc_training import BccTrainer

        if self.polarizability_type == PolarizabilityType.Element:
            from factorpol.utilities import FactorPolETBccs as this_bcc_collection
        elif self.polarizability_type == PolarizabilityType.SMIRNOFF:
            from factorpol.utilities import FactorPolSFBccs as this_bcc_collection
        else:
            raise NotImplementedError

        ret = BccTrainer.generate_charges(
            self.offmol, this_bcc_collection.recharge_collection
        )
        return Q_(ret.reshape(-1), ureg.elementary_charge)

    def free_esp_charges(self):
        """
        Generate unconstrained QM ESP derived partial charges
        :return: matrix, vector, final charges
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

    def forced_symmetry_esp_charges(self):
        """
        Generate QM ESPs derived partial charges with forced symmetry
        :return: matrix, vector, final charges
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

    def resp_style_dpolq(self, pre_charge):
        """
        Generate RESP-like partial charges in the context of direct polarization
        :param pre_charge: Unconstrained QM ESPs-derived partial charges, i.e., first stage fitting
        :return: matrix, vector, final charges
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

    def calc_Esps_dpol(self, partial_charge):
        """
        Calculate Coulomb potentials using input partial charges.
        :param partial_charge: Input partial charges
        :return: Final Coulomb potentials, Potentials generated by induced dipoles
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

    def _calc_efield(self, partial_charge):
        """
        Calculate electric fields generated by permanent partial charges.
        :param partial_charge: Input partial charges
        :return: Electric fields
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

    def calc_Esps(self, partial_charge):
        """
        Calculate Coulomb potentials generated by permanent partial charges
            $V_i = \sum\limits_{j = 1}^{n}\frac{q_j}{r_{ij}}$

        :param partial_charge: Input partial charges
        :return: Electrostatic potentials
        """
        esps = np.dot(partial_charge, self.rec_rij)

        return esps

    def calc_molecular_dipoles(self, partial_charges) -> pint.Quantity:
        """
        Calculate molecular dipole moments using permanent electrostatics
        :param partial_charges: Input partial charges
        :return: Molecular dipole moment
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

    def calc_molecular_dipoles_dpol(self, partial_charges) -> pint.Quantity:
        """
        Calculate molecular dipole moments and induced dipoles
        :param partial_charges: Input partial charges
        :return: molecular dipole moments
        """
        if isinstance(partial_charges, pint.Quantity):
            qs = partial_charges.magnitude

        efield = self._calc_efield(partial_charges)
        alphas = copy.deepcopy(self.alphas)
        alphas = np.array(alphas).reshape(-1)
        induced_dipoles = np.linalg.norm(np.multiply(efield, alphas), axis=-1).reshape(-1)
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

    def calc_Esps_mpol(self, partial_charge):
        """
        Calculate Coulomb potentials using mutual polarization
        $$\overrightarrow{\mu}_{\textrm{ind,j}} = \alpha_j \overrightarrow{E}_j$$
        Electric field by dipole
        $$\overrightarrow{E}_{\mu} =
        \frac{1}{\overrightarrow{r}^3}[(3\overrightarrow{\mu}\cdot r) r - \overrightarrow{\mu} ]$$
        :param partial_charge: Input partial charges
        :return: Final ESPs and ESPs generated by induced dipoles
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

    def _calc_efield_by_dipole(self, dipole):
        """
        Calculate electris fields generated by induced dipoles
        :param dipole: Induced dipoles
        :return: Electric field
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

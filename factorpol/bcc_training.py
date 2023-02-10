import numpy as np
from openff.recharge.aromaticity import AromaticityModel
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    BCCParameter,
    original_am1bcc_corrections,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.toolkit.topology import Molecule
from openff.recharge.utilities.toolkits import match_smirks
from openff.recharge.esp.storage import MoleculeESPRecord
from typing import List
from factorpol.utilities import flatten_a_list
from enum import Enum
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm
from factorpol.charge_training import ChargeTrainer

original_bcc_collections = original_am1bcc_corrections()
aromaticity_model = original_bcc_collections.aromaticity_model


def find_bccs(smiles: str, reference_collections: BCCCollection) -> List:
    offmol = Molecule.from_smiles(smiles)
    parameters = reference_collections.parameters
    aromaticity_model = reference_collections.aromaticity_model
    atom_aromatic, bond_aromatic = AromaticityModel.apply(offmol, aromaticity_model)
    check = lambda x: match_smirks(
        x.smirks, offmol, atom_aromatic, bond_aromatic, False
    )
    ret = [parm.smirks for parm in parameters if len(check(parm)) > 0]
    return ret


def _calc_polarization(
    worker: ChargeTrainer, alphas: Enum, coulomb14scale: float = 0.5
) -> np.ndarray:
    offmol = worker.offmol
    am1 = QCChargeGenerator.generate(
        offmol, offmol.conformers, QCChargeSettings(theory="am1")
    )
    am1 = am1.reshape(-1)
    # used if optimizing alphas and coulomb14scale
    worker.coulomb14scale = coulomb14scale
    worker.alphas = alphas
    _, ret = worker.calc_Esps_dpol(am1)
    return ret


class BccTrainer:
    def __init__(
        self,
        training_set: List[MoleculeESPRecord],
        reference_collection: BCCCollection,
        polarizability_type: Enum,
    ):
        self.bcc_collection_to_train = None
        self.bcc_parameters_to_train = None
        self.training_set = training_set
        self.reference_collection = reference_collection
        self.polarizability_type = polarizability_type
        self.charge_workers = [
            ChargeTrainer(record=r, polarizability_type=self.polarizability_type)
            for r in self.training_set
        ]

    @classmethod
    def generate_charges(cls, smiles, bcc_collection):
        offmol = Molecule.from_smiles(smiles)
        conformers = ConformerGenerator.generate(
            offmol,
            ConformerSettings(method="omega", sampling_mode="sparse", max_conformers=1),
        )
        am1 = QCChargeGenerator.generate(
            offmol,
            conformers,
            QCChargeSettings(theory="am1", sysmmetrize=False, optimize=False),
        )
        assignment_matrix = BCCGenerator.build_assignment_matrix(offmol, bcc_collection)
        pbccs = BCCGenerator.apply_assignment_matrix(
            assignment_matrix=assignment_matrix, bcc_collection=bcc_collection
        )
        ret = am1 + pbccs
        return ret

    def training(self, alphas: Enum):

        self.bcc_parameters_to_train = list(
            set(
                flatten_a_list(
                    [
                        find_bccs(c.smiles, self.reference_collection)
                        for c in self.charge_workers
                    ]
                )
            )
        )
        self.bcc_collection_to_train = BCCCollection(
            parameters=[
                BCCParameter(smirks=sm, value=0.0)
                for sm in self.bcc_parameters_to_train
            ]
        )
        generators = ESPObjective.compute_objective_terms(
            esp_records=self.training_set,
            charge_collection=QCChargeSettings(theory="am1"),
            bcc_collection=self.bcc_collection_to_train,
            bcc_parameter_keys=self.bcc_parameters_to_train,
        )
        objective_term = ESPObjectiveTerm.combine(*generators)
        dimension = objective_term.atom_charge_design_matrix.shape[0]
        polarization_objective_term = np.zeros(dimension)
        am1_polarization = [_calc_polarization(c, alphas) for c in self.charge_workers]
        # combine objective function
        for i1, m1 in enumerate(am1_polarization):
            s1 = m1.shape[0]
            if i1 == 0:
                polarization_objective_term[:s1] = m1
                s2 = s1
            else:
                polarization_objective_term[s2 : s2 + s1] = m1
                s2 += s1
        polarization_objective_term = polarization_objective_term.reshape(-1, 1)
        keys = ["x", "residuals", "rank", "singular"]

        ret = np.linalg.lstsq(
            a=objective_term.atom_charge_design_matrix,
            b=objective_term.reference_values - polarization_objective_term,
            rcond=None,
        )
        results = {k: v for k, v in zip(keys, ret)}
        results["bcc_parameters"] = {
            b: p[0] for b, p in zip(self.bcc_parameters_to_train, results["x"])
        }
        return results

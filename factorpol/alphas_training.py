"""
This module provides functionalities to derive and obtain typed polarizabilities from QM reference data.
"""

import os

import pint
import ray
import shutil

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
from typing import List
from openff.recharge.esp.storage import MoleculeESPRecord
from enum import Enum
from factorpol.charge_training import ChargeTrainer
import numpy as np
import copy
from factorpol.utilities import flatten_a_list
from factorpol.utilities import PolarizabilityType, StorageHandler, Polarizability, calc_rrmse
from scipy.optimize import minimize
from factorpol.qm_worker import rebuild_molecule
from openff.toolkit import ForceField

# elementary_charge/bohr to kcal/mol/elementary_charge
au_to_kcal = 633.0917033332278


class AlphaWorker(ChargeTrainer):
    def __init__(
            self,
            record: MoleculeESPRecord,
            polarizability_type: Enum,
            ff: ForceField,
            parameters_path: str = "default",
            coulomb14scale: float = 0.5,
    ):
        """
        A class used to optimize polarizabilities
        :param record: QM reference record
        :param polarizability_type: Polarizability Typing scheme
        :param parameters_path: Output path of updated parameters
        :param coulomb14scale: Coulomb14 scaling factor
        """
        super().__init__(
            record=record,
            polarizability_type=polarizability_type,
            ff=ff
        )
        if (
                self.polarizability_type == PolarizabilityType.Element
                and parameters_path == "default"
        ):
            from factorpol.utilities import ETPol as alphas

        elif (
                self.polarizability_type == PolarizabilityType.SMIRNOFF
                and parameters_path == "default"
        ):
            from factorpol.utilities import SmirnoffPol as alphas

        else:
            self.polarizability_type = polarizability_type
            alphas = Polarizability(data_source=parameters_path, typing_scheme=self.polarizability_type)

        self.alphas = alphas
        self.coulomb14scale = coulomb14scale
        self.perturb_dipole = record.esp_settings.perturb_dipole

    @property
    def vdiff(self):
        return self._vdiff

    @vdiff.setter
    def vdiff(self, value: np.ndarray):
        self._vdiff = value


def _update_workers(
        workers: List[AlphaWorker], parameters_path: str, polarizability_type: Enum, coulomb14scale: float = 0.5
):
    """
    Method to update charge workers for next optimization iteration
    :param workers: AlphaWorker Object
    :param parameters_path: Output path for polarizability parameters
    :param polarizability_type: Polarizability typing scheme
    :param coulomb14scale: coulomb14 scaling factor
    :return: Updated AlphaWorker object
    """
    alphas = Polarizability(
        data_source=parameters_path, typing_scheme=polarizability_type
    )
    for w in workers:
        w.alphas = alphas
        w.coulomb14scale = coulomb14scale
    return workers


class AlphasTrainer:
    def __init__(
            self,
            workers: List[AlphaWorker],
            polarizability_type: Enum,
            prior: Enum,
            working_directory: str = os.path.join(os.getcwd(), "data_alphas"),
    ):
        """
        Top level optimizer to train polarizability parameters
        :param workers:
        :param polarizability_type:
        :param prior:
        :param working_directory:
        """
        self.coulomb14scale = None
        self.alphas_path = None
        self.polarizability_type = polarizability_type
        self.working_directory = working_directory
        self.iteration = 0
        if self.polarizability_type == PolarizabilityType.Element:
            from factorpol.utilities import ETPol
            self.base = ETPol.data
        # elif self.polarizability_type == PolarizabilityType.SMIRNOFF:
        #     from factorpol.utilities import SmirnoffPol
        #     self.base = SmirnoffPol.data
        else:
            self.polarizability_type = polarizability_type
            self.base = prior.data
            # raise NotImplementedError
        self.parameter_type_to_train = prior.parameters.keys()
        self.prior = [v.magnitude for v in prior.parameters.values()]

        if os.path.exists(self.working_directory):
            print("Path exists, deleting")
            shutil.rmtree(self.working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        self.workers = workers

    def worker(self, input_data):
        """
        A method to calculate loss function
        :param input_data: Input polarizability
        :return: Output objective
        """
        self.iteration += 1
        self.alphas_path = os.path.join(self.working_directory, f"alpha_{self.iteration:03d}.log")

        for k, v in zip(self.parameter_type_to_train, input_data):
            self.base.loc[k, "Polarizability (angstrom**3)"] = v

        self.base.to_csv(self.alphas_path)

        workers = _update_workers(
            workers=self.workers,
            parameters_path=self.alphas_path,
            polarizability_type=self.polarizability_type
        )

        loss = [AlphasTrainer._calc_loss.remote(w) for w in workers]
        ret = np.mean(ray.get(loss))
        os.system(f"echo {ret} >> {os.path.join(self.working_directory, 'Loss.log')}")
        return ret

    def optimize(self, bounds, num_cpus=8):
        """
        Use Ray and Scipy optimizer to optimize polarizabilities
        :param bounds:
        :param num_cpus:
        :return:
        """
        ray.shutdown()
        ray.init(num_cpus=num_cpus, num_gpus=0)
        x0 = self.prior
        # a simple boundary
        # y = lambda x: (x/2, x*2)
        # bounds = tuple([y(x) for x in x0])
        res = minimize(self.worker, x0=x0, method="Nelder-Mead", bounds=bounds)
        ray.shutdown()
        return res

    @staticmethod
    def _calc_Esps_mu(worker: AlphaWorker) -> np.ndarray:
        """
        Calculate MM ESPs using mutual polarizabilities
        :param worker: AlphaWorker
        :return: MM ESPs
        """
        external_field = worker.perturb_dipole
        # cast a local electric field matrix
        efield = np.full_like(a=np.zeros((worker.natoms, 3)), fill_value=external_field)
        esps = np.zeros(worker.npoints)
        alphas = copy.deepcopy(worker.alphas)
        for j in range(worker.natoms):
            for i in range(worker.npoints):
                esps[i] += alphas[j] * np.dot(
                    efield[j], worker.r_ij[j, i] * worker.r_ij3[j, i]
                )
        return esps

    @staticmethod
    @ray.remote(num_cpus=1)
    def _calc_loss(worker: AlphaWorker) -> float:
        """
        Method to calculate objective function for one worker
        :param worker: AlphaWorker
        :return: Loss
        """
        calced = AlphasTrainer._calc_Esps_mu(worker)
        ref = worker.vdiff
        loss = calc_rrmse(calced, ref)  # rrmse
        return loss


class AlphaData:
    def __init__(
            self, database_name: str, dataset: List[str], polarizability_type: Enum, parameter_path, ff, num_cpus: int = 8
    ):
        """
        A class to prepare reference data to derive polarizabilities
        :param database_name: SQL dataset name
        :param dataset: Training molecule
        :param polarizability_type: Polarizability typing scheme
        :param num_cpus: Number of CPUs available to prepare dataset
        """
        self.database_name = database_name
        self.dataset = dataset
        self.workers = []
        self.polarizability_type = polarizability_type

        ray.shutdown()
        ray.init(num_cpus=num_cpus)

        ret = [
            create_worker.remote(database_name, molecule=mol, polarizability_type=self.polarizability_type,
                                  ff=ff, parameters_path=parameter_path)
            for mol in self.dataset
        ]
        workers = ray.get(ret)
        self.workers = flatten_a_list(workers)

        ray.shutdown()


@ray.remote(num_cpus=1)
def create_worker(
        database_name: str,
        molecule: str,
        polarizability_type: Enum,
        ff: ForceField,
        parameters_path: str = "default",
        coulomb14scale: float = 0.5,
) -> List[AlphaWorker]:
    """
    A function to create an AlphaWorker using QM reference data
    :param database_name: SQL database name
    :param molecule: one molecule
    :param polarizability_type: Polarizability Typing scheme
    :param parameters_path: Polarizability parameters path
    :param coulomb14scale: Coulomb14 scaling factor
    :return: AlphaWorker
    """
    workers = []

    store = StorageHandler()
    my_session = store.session(database_name)

    records_dict = rebuild_molecule(my_session=my_session, molecule=molecule)

    for conf_idx, records in records_dict.items():
        base = None
        this_conf = []
        for r in records:
            worker = AlphaWorker(
                record=r,
                polarizability_type=polarizability_type,
                ff=ff,
                parameters_path=parameters_path,
                coulomb14scale=coulomb14scale,
            )
            if np.allclose(np.zeros(3), r.esp_settings.perturb_dipole):
                base = copy.deepcopy(worker)
                base.vdiff = np.zeros(base.npoints)
            else:
                this_conf.append(worker)

        for w in this_conf:
            w.vdiff = w.esp_values - base.esp_values

        workers.extend(this_conf)
        #workers.append(base)
    return workers

"""
This module provides functionalities to derive and obtain typed polarizabilities from QM reference data.
"""

import copy
import os
import shutil
from typing import List

import numpy as np
import pint
import ray
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit import ForceField
from scipy.optimize import minimize

from factorpol.charge_training import ChargeTrainer
from factorpol.qm_worker import rebuild_molecule
from factorpol.utilities import (
    calc_rrms,
    flatten_a_list,
    Polarizability,
    StorageHandler,
)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# elementary_charge/bohr to kcal/mol/elementary_charge
au_to_kcal = 633.0917033332278


class AlphaWorker(ChargeTrainer):
    """
    A class used to derive polarizability from QM ESPs.

    Parameters
    ----------
    record: MoleculeESPRecord
        QM reference record

    off_forcefield: ForceField
         OpenFF force field to handle SMIRNOFF typing of polarizabilities

    polarizability: Polarizability
        Input polarizabilities

    coulomb14scale: float
        Scaling factor to scale Coulomb 1-4 interactions

    """

    def __init__(
        self,
        record: MoleculeESPRecord,
        off_forcefield: ForceField,
        polarizability: Polarizability,
        coulomb14scale: float = 0.5,
    ):
        super().__init__(
            record=record,
            off_forcefield=off_forcefield,
            polarizability=polarizability,
            coulomb14scale=coulomb14scale,
        )

        # self.alphas = polarizability
        self.coulomb14scale = coulomb14scale
        self.perturb_dipole = record.esp_settings.perturb_dipole

    @property
    def vdiff(self):
        return self._vdiff

    @vdiff.setter
    def vdiff(self, value: np.ndarray):
        self._vdiff = value


def _update_workers(
    workers: List[AlphaWorker], parameters_path: str, coulomb14scale: float = 0.5
):
    """
    Method to update charge workers for next optimization iteration

    Parameters
    ----------
    workers: List[AlphaWorker]
        A list of AlphaWorker

    parameters_path: str
        Path to newly solved polarizabilities

    coulomb14scale: float
        Scaling factor of Coulomb 1-4 interactions.
        Normally set as constant but it could be optimized.

    Returns
    -------
    AlphaWorker
        Returns updated AlphasWorker

    """

    pols = Polarizability(data_source=parameters_path)
    for w in workers:
        w.alphas = [
            pols.parameters[p].to("bohr**3").magnitude for p in w.smirnoff_patterns
        ]
        # w.coulomb14scale = coulomb14scale # remove comment if optimizing
    return workers


class AlphasTrainer:
    """
    Top level optimizer to train polarizability parameters

    Parameters
    ----------
    workers: List[AlphaWorker]
        A list of AlphaWorker

    prior: Polarizability
        Initial polarizability

    working_directory: str
        The path to the working directory

    """

    def __init__(
        self,
        workers: List[AlphaWorker],
        prior: Polarizability,
        working_directory: str = os.path.join(os.getcwd(), "data_alphas"),
    ):
        self.coulomb14scale = None
        self.alphas_path = None
        self.working_directory = working_directory
        self.iteration = 0
        self.base = prior.data
        self.parameter_type_to_train = prior.parameters.keys()
        self.prior = [v.magnitude for v in prior.parameters.values()]

        if os.path.exists(self.working_directory):
            print("Path exists, deleting")
            shutil.rmtree(self.working_directory)
        os.makedirs(self.working_directory, exist_ok=True)
        self.workers = workers

    def worker(self, input_data: np.ndarray):
        """
        A method to compute the loss

        Parameters
        ----------
        input_data: ndarray
            Polarizability solved from the previous optimization

        Returns
        -------
            Returns the optimizer output object.

        """

        self.iteration += 1
        self.alphas_path = os.path.join(
            self.working_directory, f"alpha_{self.iteration:03d}.log"
        )

        for k, v in zip(self.parameter_type_to_train, input_data):
            self.base.loc[k, "Polarizability (angstrom**3)"] = v

        self.base.to_csv(self.alphas_path)

        workers = _update_workers(
            workers=self.workers,
            parameters_path=self.alphas_path,
        )

        loss = [AlphasTrainer._calc_loss.remote(w) for w in workers]
        ret = np.mean(ray.get(loss))
        os.system(f"echo {ret} >> {os.path.join(self.working_directory, 'Loss.log')}")
        return ret

    def optimize(self, bounds, num_cpus=8):
        """
        Distribute the optimization process with `Ray`

        Parameters
        ----------
        bounds: tuple
            The bounds of each polarizability type

        num_cpus: int
            The number of CPUs available for optimization

        Returns
        -------
            Returns the result object of optimizer.
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
        Calculate the MM ESPs on grid points using mutual polarization

        Parameters
        ----------
        worker: AlphaWorker
            The AlphaWorker that contains all ESP-fitting related data

        Returns
        -------
        ndarray
            Returns the computed ESPs on grid points

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
        Calculate the lost function (RRMS) for one worker

        Parameters
        ----------
        worker: AlphasWorker
            The AlphaWorker that contains all ESP-fitting related data

        Returns
        -------
        float
            Returns the Loss (RRMS) value

        """

        calced = AlphasTrainer._calc_Esps_mu(worker)
        ref = worker.vdiff
        loss = calc_rrms(calced, ref)  # rrmse
        return loss


class AlphaData:
    """
    A class to prepare reference QM ESPs for optimization of polarizability

    Parameters
    ----------
    database_name: str
        The name of database to query

    dataset: List[str]
        A list of molecules to query

    off_forcefield: ForceField
        An OpenFF Force Field to specify SMIRNOFF patterns

    polarizability: Polarizability
        A polarizability library

    num_cpus: int
        The number of process to initialize to generate relevant data

    """

    def __init__(
        self,
        database_name: str,
        dataset: List[str],
        off_forcefield: ForceField,
        polarizability: Polarizability,
        num_cpus: int = 8,
    ):
        self.database_name = database_name
        self.dataset = dataset
        self.workers = []

        ray.shutdown()
        ray.init(num_cpus=num_cpus)

        ret = [
            create_worker.remote(
                database_name,
                molecule=mol,
                polarizability=polarizability,
                off_forcefield=off_forcefield,
                coulomb14scale=0.5,
            )
            for mol in self.dataset
        ]
        workers = ray.get(ret)
        self.workers = flatten_a_list(workers)

        ray.shutdown()


@ray.remote(num_cpus=1)
def create_worker(
    database_name: str,
    molecule: str,
    polarizability: Polarizability,
    off_forcefield: ForceField,
    coulomb14scale: float = 0.5,
) -> List[AlphaWorker]:
    """
    This is a function to gather all necessary information to optmize polarizability

    Parameters
    ----------
    database_name: str
        The name of dataset to query

    molecule: str
        The SMILES to look for.

    polarizability: Polarizability
        A polarizability library

    off_forcefield: ForceField
        OpenFF ForceField to label molecules with SMIRNOFF patterns.

    coulomb14scale: float
        Scaling factor of Coulomb 1-4 interactions.
        Normally set as constant but it could be optimized.

    Returns
    -------
    List[AlphaWorker]
        Returns a list of prepared AlphaWorkers.

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
                off_forcefield=off_forcefield,
                polarizability=polarizability,
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
    return workers

"""
This module provides functionalities to derive and obtain typed polarizabilities from QM reference data.
"""

import copy
import os
import shutil
from typing import List
import logging
import numpy as np
import pint
import ray
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.toolkit import ForceField
from scipy.optimize import minimize, nnls
from collections import defaultdict

from factorpol.charge_training import ChargeTrainer
from factorpol.qm_worker import rebuild_molecule
from factorpol.utilities import (
    calc_rrms,
    flatten_a_list,
    Polarizability,
    StorageHandler,
    pair_equivalent,
)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

logger = logging.getLogger(__name__)

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
            logger.warning("Path exists, deleting")
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

@ray.remote(num_cpus=1)
def fit_alphas(worker: AlphaWorker, global_opt=False) -> np.ndarray:
    r"""
    Fit the polarizability of a molecule to reference QM ESPs
    This is a remote function to be called by ray
    Atomic units

    .. math:: \sum_{j = 1}^{n}\sum_{i = 1}^{m}\frac{\alpha_j \mathrm{E^2}}{r_{ij}^3r_{ik}} = \sum_{i = 1}^{m}\frac{V_i \mathrm{E}}{r_{ik}}

    Parameters
    -----------
    worker: AlphaWorker
        The AlphaWorker that contains all ESP-fitting related data

    global_opt: bool
        Whether to perform global optimization

    Returns
    -----------
    ndarray
        Returns the fitted polarizability alphas if global_opt is False
        Returns built matrix, vector, and polarizability types if global_opt is True

    """
    natoms = worker.natoms
    external_field = worker.perturb_dipole
    # cast a local electric field matrix
    efield = np.full_like(a=np.zeros((natoms, 3)), fill_value=external_field)
    vdiff = worker.vdiff
    
    # find same polarizability types to constraint them to be the same
    tmp1 = defaultdict(list)
    for idx1, rank in enumerate(worker.smirnoff_patterns):
        tmp1[rank].append(idx1)

    tmp2 = []
    tmp3 = {}

    for k, v in tmp1.items():
        tmp3[k] = v[0]
        if len(v) > 1:
            tmp2.append([[v[i], v[i + 1]] for i in range(len(v) - 1)])

    alphas_pairs = np.concatenate(tmp2)

    ndim = worker.natoms + len(alphas_pairs)
    
    # Distance matrix
    D_ij = np.multiply(worker.r_ij, worker.r_ij3)

    matrix = np.zeros((ndim, ndim, 3))
    for k in range(natoms):
        for j in range(natoms):
            for i in range(worker.npoints):
                matrix[k, j] += (
                        np.square(external_field) * D_ij[k, i] * D_ij[j, i]
                )
    
    # force some polarizability types to have the same values
    matrix = np.linalg.norm(matrix, axis=-1)
    for idx, pair in enumerate(alphas_pairs):

        matrix[natoms + idx, pair[0]] = 1.0
        matrix[natoms + idx, pair[1]] = -1.0
        matrix[pair[0], natoms + idx] = 1.0
        matrix[pair[1], natoms + idx] = -1.0

    vector = np.zeros((ndim, 3))

    for k in range(natoms):
        for i in range(worker.npoints):
            vector[k] += external_field * D_ij[k, i] * vdiff[i]

    vector = np.linalg.norm(vector, axis=-1)
    if global_opt == True:
        return matrix[:natoms, :natoms], vector[:natoms], worker.smirnoff_patterns
    
    else:
        ret = np.linalg.solve(matrix, vector)
        return ret[:natoms]


def optimize_alphas(worker_list: List[AlphaWorker], solved=True) -> np.ndarray:
    r"""
    A function to optimize the polarizability of a dataset to reference QM ESPs
    Atomic units

    Maths:
    .. math::
          \chi^2 = \sum\limits_{k=1}^{N_\textrm{conf}} \sum\limits_{l=1}^{6}  \sum\limits_{i=1}^{m}  \left( V_\textrm{diff,ikl} -\sum\limits_{j=1}^{n_k}\frac{\vect{\mu}_{\textrm{ind,jl}}\vect{r}_{ij}}{r_{ij}^3} \right)^2

    
    Parameters
    -----------
    worker_list: List[AlphaWorker]

    solved: bool
        Whether the polarizability is solved or not

    Returns
    ----------- 
    ndarray, ndarray, ndarray
        Returns matrix, vector, and polarizability types
        Returns the fitted polarizability alphas and objectives if solved is True
    """
    
    workers = [fit_alphas.remote(w, global_opt=True) for w in worker_list]

    # Get results from ray
    ret = ray.get(workers)
    
    # Split the returns to matrix, vector, and polarizability types
    a_lst = [r[0] for r in ret]
    b_lst = [r[1] for r in ret]
    pol_lst = [r[-1] for r in ret]
    
    # Calculate and pair the same polarizabilities
    pol_lst_flatten = flatten_a_list(pol_lst)
    pairs = pair_equivalent(pol_lst_flatten)
    n_same_alphas = len(pairs)
    ndim = np.sum([m.natoms for m in worker_list]) + n_same_alphas

    # Initializing a new matrix to do the optimization
    final_a = np.zeros((ndim, ndim))

    for idx, this_a in enumerate(a_lst):
        natoms = this_a.shape[0]
        if idx == 0:
            final_a[:natoms, :natoms] = this_a
        else:
            final_a[natoms*(idx-1)+natoms:natoms*idx+natoms, natoms*(idx-1)+natoms:natoms*idx+natoms] = this_a

    # Apply same alphas restraints

    for idx, pair in enumerate(pairs):
        this_idx = (ndim - n_same_alphas) + idx
        final_a[this_idx, pair[0]] = 1.0
        final_a[this_idx, pair[1]] = -1.0

        final_a[pair[0], this_idx] = 1.0
        final_a[pair[1], this_idx] = -1.0

    final_b = np.append(np.concatenate(b_lst), np.zeros(n_same_alphas))

    if solved == True:
        ret = nnls(final_a, final_b)
        dt = {k: Q_(v, 'a0**3').to('angstrom**3') for k, v in zip(pol_lst_flatten, ret[0][:len(pol_lst_flatten)])}
        return dt, ret[-1]

    else:
        return final_a, final_b, set(pol_lst_flatten)

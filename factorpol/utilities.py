"""
    This module contains useful functions to use with Factor-Pol model

"""
import enum
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import pint
from numpy import ndarray
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCParameter,
    original_am1bcc_corrections,
)
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.esp.storage.db import DBBase, DBMoleculeRecord
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from pkg_resources import resource_filename
from rdkit import Chem
from sqlalchemy import create_engine, select
from sqlalchemy.orm import session, sessionmaker
from sqlalchemy_utils import create_database, database_exists

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

original_bcc_collections = original_am1bcc_corrections()
aromaticity_model = original_bcc_collections.aromaticity_model
ff = ForceField("openff-2.0.0.offxml")


def pair_equivalent(pattern: List) -> ndarray:
    """
    A function to pair related patterns together for use as constraints

    Parameters
    ----------
    pattern: List
        A list of patterns, could be elements, SMIRNOFF patterns

    Returns
    -------
    ndarry
        Return pairs of related patterns in a nested numpy ndarry.

    """
    tmp1 = defaultdict(list)
    for idx1, p in enumerate(pattern):
        tmp1[p].append(idx1)

    tmp2 = []
    for key, v in tmp1.items():
        n = len(v)
        if n > 1:
            tmp2.append([[v[i], v[i + 1]] for i in range(n - 1)])
    if len(tmp2) == 0:
        ret = []
    else:
        ret = np.concatenate(tmp2)
    return ret


def canonical_ranking(rdmol: Chem.rdchem.Mol) -> List:
    """
    A function to calculte canonical ranking for forced symmetry using RDKit

    Parameters
    ----------
    rdmol: Chem.rdchem.Mol
        A rdkir molecule object

    Returns
    -------
    List
        A list of atomic features based on the canonical ranking of all atoms

    """

    ret = list(Chem.rdmolfiles.CanonicalRankAtoms(rdmol, breakTies=False))
    return ret


def smirnoff_labels(offmol: Molecule, off_forcefield: ForceField) -> List:
    """
    A function to label OpenFF molecule objecit with SMIRNOFF patternes specified in the input OpenFF ForceField object.

    Parameters
    ----------
    offmol: Molecule
        The input molecule to label

    off_forcefield: ForceField
        The input openff ForceField with SMIRNOFF patterns to label atoms in molecule.

    Returns
    -------
    List
        Return a list of SMIRNOFF patterns associated with atoms in molecule

    """

    off_topology = Topology.from_molecules(offmol)
    parameters_list = off_forcefield.label_molecules(off_topology)[0]
    ret = [v._smirks for _, v in parameters_list["vdW"].items()]
    return ret


def flatten_a_list(nest_list: List) -> List:
    """
    A handy funtion to flatten a nested list

    Parameters
    ----------
    nest_list: List
        A nested list that needed to be flatten into a 1-D list

    Returns
    -------
    List
        Return a 1-D list

    """
    return [item for sublst in nest_list for item in sublst]


def coulomb_scaling(rdmol: Chem.rdchem.Mol, coulomb14scale: float = 0.5) -> ndarray:
    """
    A function to create scaling matrix for scaling the 1-4 interactions in Coulomb interactions

    Parameters
    ----------
    rdmol: Chem.rdchem.Mol
        An input rdkit molecule used for specifying connectivity

    coulomb14scale: float
        The coulomb14 scaling factor, default value is 0.5. Commonly used value includes 0.83333

    Returns
    -------
    ndarray
        Returns a numpy ndarray as scaling matrix for using in scaling Coulomb interactions.
        This scaling matrix excludes all 1-2, 1-3 interactions and scales 1-4 by coulomb14scale factor.

    """

    natom = rdmol.GetNumAtoms()
    # initializing arrays
    bonds = []
    bound12 = np.zeros((natom, natom))
    bound13 = np.zeros((natom, natom))
    scaling_matrix = np.ones((natom, natom))

    for bond in rdmol.GetBonds():
        b = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()
        bonds.append([b, e])

    # Find 1-2 scaling_matrix
    for pair in bonds:
        bound12[pair[0], pair[1]] = 12.0
        bound12[pair[1], pair[0]] = 12.0

    # Find 1-3 scaling_matrix
    b13_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for idx, j in enumerate(b12_idx):
            for k in b12_idx[idx + 1 :]:
                b13_pairs.append([j, k])
    for pair in b13_pairs:
        bound13[pair[0], pair[1]] = 13.0
        bound13[pair[1], pair[0]] = 13.0

    # Find 1-4 scaling_matrix
    b14_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for j in b12_idx:
            b122_idx = np.nonzero(bound12[j])[0]
            for k in b122_idx:
                for j2 in b12_idx:
                    if k != i and j2 != j:
                        b14_pairs.append([j2, k])

    # Assign coulomb14scaling factor
    for pair in b14_pairs:
        scaling_matrix[pair[0], pair[1]] = coulomb14scale
        scaling_matrix[pair[1], pair[0]] = coulomb14scale

    # Exclude 1-2, 1-3 interactions
    for pair in bonds:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    for pair in b13_pairs:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    # Fill 1-1 with zeros
    np.fill_diagonal(scaling_matrix, 0)

    return scaling_matrix


class StorageHandler:
    """
    This is a handler to interact with data stored in PostgreSQL database.

    Parameters
    ----------
    port: int
        The port where PostgreSQL server is running. Default is `5432`.

    url: str
        The url in form of a string which contains the path to a running PostgreSQL server.

    local_path: str
        A local path to store temporary data.
        Default is a directory named `tmp_storage` at current working directory.

    """
    def __init__(
            self,
            port: str = "5432",
        url: str = "postgresql://localhost:",
        local_path: str = os.path.join(os.getcwd(), "tmp_storage"),
    ):

        self.port = port
        self.url = url
        self.postgres_prefix = f"{self.url}{self.port}"
        self.local_path = local_path
        self.start_err = None
        self.start_out = None

        if os.path.exists(self.local_path):
            pass
            # shutil.rmtree(self.local_path)
        else:
            os.makedirs(self.local_path)

    def start(self):
        """
        This method will start a server if currently there is no active server running on specified url.

        Returns
        -------
        str
            Return output and/or error messages

        """

        _ = subprocess.Popen(
            ["initdb", f"{self.local_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(15)

        if self.port != "5432":
            with open(os.path.join(self.local_path, "postgresql.conf"), "r") as f:
                conf = f.read()
            conf = conf.replace("#port = 5432", f"port = {self.port}")
            with open(os.path.join(self.local_path, "postgresql.conf"), "w") as f:
                f.write(conf)

        ret = subprocess.Popen(
            [
                "pg_ctl",
                "-D",
                f"{self.local_path}",
                "-l",
                f"{self.local_path}/postgresql.log",
                "start",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.start_err, self.start_out = ret.communicate()

        return self.start_out, self.start_err

    def stop(self):
        """
        This method will stop the server associated to this storage handler.

        Returns
        -------
        str
            Returns output and/or error messages

        """

        ret = subprocess.Popen(
            [
                "pg_ctl",
                "-D",
                f"{self.local_path}",
                "stop",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stop_err, stop_out = ret.communicate()

        return stop_out, stop_err

    def session(self, database_name: str) -> session.Session:
        """
        This is a handy method to create a sqlalchemy session for input database to use in querying data.

        Parameters
        ----------
        database_name: str
            The name of database to query

        Returns
        -------
        session.Session
            Returns a working Session to use in querying data.

        """

        this_database = f"{self.postgres_prefix}/{database_name}"
        my_engine = create_engine(this_database)
        if database_exists(my_engine.url):
            print(f"Found database {my_engine.url}")
        else:
            print(f"Creating new database at {my_engine.url}")
            create_database(my_engine.url)
            DBBase.metadata.create_all(my_engine)
        my_session = sessionmaker(bind=my_engine, autoflush=False, autocommit=False)
        my_session = my_session()

        return my_session


def calc_rrms(calc: ndarray, ref: ndarray):
    r"""
    A function to calculate relative root mean squared error, RRMS error, unit less

    .. math::
        RRMS =\sqrt{\frac{1}{N}\frac{\sum\limits_{i=1}^{N}(V_{qm,i}-V_{calc, i})^2}{\sum\limits_{i=1}^{N}(V_{qm, i})^2}}

    Parameters
    ----------
    calc: ndarray
        Calculated data

    ref: ndarray
        Reference data

    Returns
    -------
    float
        Returns the RRMS error value

    """

    ndata = len(calc)
    ret = np.sqrt((np.sum(np.square(calc - ref)) / np.sum(np.square(calc))) / ndata)
    return ret


def calc_rmse(calc: ndarray, ref: ndarray):
    r"""
    A function to calculate  root mean squared error, RMSE, unit is the same as input data

    Parameters
    ----------
    calc: ndarray
        Calculated data

    ref: ndarray
        Reference data

    Returns
    -------
    float
        Returns the RMSE value

    """

    ret = np.sqrt(np.mean(np.square(calc - ref)))
    return ret


def retrieve_records(
    my_session: session.Session,
    dataset: List = [],
    sqlite_path: str = os.path.join(os.getcwd(), "tmp.sqlite"),
) -> Dict:
    """
    A function to retrieve data from the input session and create `MoleculeESPRecords` for use of
    polarizability or charge fitting.

    Parameters
    ----------
    my_session: session.Session
        A session associated to the PostgreSQL database to look for data.

    dataset: List
        A list of SMILES string of molecules to look for.

    sqlite_path: str
        The path to create and storage a local copy of MoleculeESPRecords.

    Returns
    -------
    dict
        Returns a dictionary of retrieved records.
        SMILES string as key, MoleculeESPRecord as value.

    """

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    else:
        pass
    tmp = MoleculeESPStore(sqlite_path)
    if len(dataset) > 0:
        db_records = [
            my_session.scalars(
                select(DBMoleculeRecord).where(DBMoleculeRecord.smiles == smi)
            ).all()
            for smi in dataset
        ]
        db_records = flatten_a_list(db_records)
    else:
        db_records = my_session.scalars(select(DBMoleculeRecord)).all()

    models = tmp._db_records_to_model(db_records)
    _ = [tmp.store(m) for m in models]
    smiles = tmp.list()

    ret = dict(zip(smiles, models))

    return ret


@dataclass
class Polarizability:
    """
    A dataclass to read/write polarizability parameters

    Parameters
    ----------
    data_source: str
        The path of a `.csv` file which stores all polarizabilities.

    Examples
    --------
    ``DefaultPol = Polarizability(data_source=resource_filename("factorpol", os.path.join("data", "alphas.example.csv")))``

    """
    data_source: str = resource_filename(
        "factorpol", os.path.join("data", "alphas.example.csv")
    )

    @property
    def data(self) -> pd.core.frame.DataFrame:
        """
        Store polarizability parameters ad pandas DataFrame

        Returns
        -------
        pd.core.frame.DataFrame
            Stored polarizability parameters as a pandas.DataFrame

        """
        dt = pd.read_csv(self.data_source, index_col="Type")
        return dt

    @property
    def parameters(self) -> Dict:
        """
        Extra types and polarizabilities and store in a dictionary for easy parameterization.

        Returns
        -------
        Dict
            A dictionary of polarizabilities.

        """
        pdt = self.data.dropna()
        ret = {
            k: Q_(v, "angstrom**3")
            for k, v in zip(pdt.index, pdt["Polarizability (angstrom**3)"])
        }
        return ret


@dataclass
class BondChargeCorrections:
    """
    A dataclass to read/write bond charge correction parameters for generating AM1-BCC-dPol charges

    Parameters
    ----------
    data_source: str
        The path of a `.csv` file which stores all bond charge correction parameters.

    Examples
    --------
    ``DefaultBccs = BondChargeCorrections(data_source=resource_filename("factorpol", os.path.join("data", "bcc_dPol.csv")))``

    """
    data_source: str = resource_filename(
        "factorpol", os.path.join("data", "bcc_dPol.csv")
    )

    @property
    def data(self) -> pd.core.frame.DataFrame:
        """
        Store BCC parameters as a pandas.DataFrame

        Returns
        -------
        pd.core.frame.DataFrame
            Stored BCC parameters ad pandas DataFrame

        """
        dt = pd.read_csv(self.data_source, index_col="BCC SMIRKS")
        return dt

    @property
    def parameters(self) -> Dict:
        """
        Extract types and BCC parameters and store in a dictionary for easy parameterization.

        Returns
        -------
        Dict
            A dictionary of BCCs.

        """
        ret = {k: v for k, v in zip(self.data.index, self.data["BCC value"])}
        return ret

    @property
    def recharge_collection(self) -> BCCCollection:
        """
        Create an `openff-recharge` BCC collection to generate AM1-BCC-dPol partial charges

        Returns
        -------
        BCCCollection
            Returned BCCCollection

        """
        ret = BCCCollection(
            parameters=[
                BCCParameter(smirks=sm, value=float(vs))
                for sm, vs in self.parameters.items()
            ]
        )
        ret.aromaticity_model = aromaticity_model
        return ret

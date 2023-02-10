import enum
import os
import shutil
import subprocess
import time
from collections import defaultdict
from typing import List, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from rdkit import Chem
from sqlalchemy import create_engine, select
from openff.recharge.esp.storage.db import DBBase
from sqlalchemy.orm import sessionmaker, session
from sqlalchemy_utils import create_database, database_exists, drop_database
from openff.recharge.esp.storage import MoleculeESPStore
from openff.recharge.esp.storage.db import DBMoleculeRecord
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCParameter,
    original_am1bcc_corrections,
)
from dataclasses import dataclass
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
from pkg_resources import resource_filename
from enum import Enum

original_bcc_collections = original_am1bcc_corrections()
aromaticity_model = original_bcc_collections.aromaticity_model
ff = ForceField("openff-2.0.0.offxml")


def pair_equivalent(pattern: List) -> ndarray:
    """
    :rtype: object
    :param pattern: A list of patterns that needs to be paired
    :return: A list of paired patterns
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
    :param rdmol: RDK molecule
    :return: A list of atomic features based the canonical ranking
    """
    ret = list(Chem.rdmolfiles.CanonicalRankAtoms(rdmol, breakTies=False))
    return ret


def smirnoff_labels(offmol: Molecule, off: ForceField) -> List:
    """
    :param offmol: OpenFF Molecule
    :param off: Open Force Field force field
    :return: a list smirks pattern by van der Waals parameter types
    """
    off_topology = Topology.from_molecules(offmol)
    parameters_list = off.label_molecules(off_topology)[0]
    ret = [v._smirks for _, v in parameters_list["vdW"].items()]
    return ret


def flatten_a_list(nest_list: List) -> List:
    """
    Handy function to flatten a nested list
    """
    return [item for sublst in nest_list for item in sublst]


def coulomb_scaling(rdmol: Chem.rdchem.Mol, coulomb14scale: float = 0.5) -> ndarray:
    """
    :param rdmol: RDKit Molecule
    :param coulomb14scale: The scaling factor of coulomb interactions, default is 0.5
    :return: A scaling_matrix matrix used for computing electric field by fixed point charges
             Exclude 1-2, 1-3, and scale 1-4 by coulomb14scale factor
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
            for k in b12_idx[idx + 1:]:
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

        return self.start_err, self.start_out

    def stop(self):

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

        stop_err, start_out = ret.communicate()

        return stop_err, start_out

    def session(self, database_name: str) -> session.Session:
        """
        :param database_name: Name of database.
        :return: Return a session to interact with the database
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


def calc_rrmse(calc, ref):
    """
    $$RRMSE=\sqrt{\frac{1}{N}\frac{\sum\limits_{i=1}^{N}(V_{qm,i}-V_{calc, i})^2}{\sum\limits_{i=1}^{N}(V_{qm, i})^2}}$$
    unit less
    """
    ndata = len(calc)
    ret = np.sqrt((np.sum(np.square(calc - ref)) / np.sum(np.square(calc))) / ndata)
    return ret


def calc_rmse(calc, ref):
    """
    :param calc:
    :param ref:
    :return:
    $$RMSE=\sqrt{\frac{1}{N}\sum\limits_{i=1}^{N}(V_{qm,i}-V_{calc, i})^2}$$
    """
    ret = np.sqrt(np.mean(np.square(calc - ref)))
    return ret


def retrieve_records(
        my_session: session.Session,
        dataset: List = [],
        sqlite_path: str = os.path.join(os.getcwd(), "tmp.sqlite"),
) -> Tuple[List, List]:
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

    return models, smiles


class PolarizabilityType(Enum):
    Element: Literal = "Element"
    SMIRNOFF: Literal = "SMIRNOFF"


@dataclass
class Polarizability:
    data_source: str
    typing_scheme: Enum

    @property
    def data(self) -> pd.core.frame.DataFrame:
        dt = pd.read_csv(self.data_source, index_col="Type")
        return dt

    @property
    def parameters(self) -> Dict:
        pdt = self.data.dropna()
        ret = {
            k: Q_(v, "angstrom**3")
            for k, v in zip(pdt.index, pdt["Polarizability (angstrom**3)"])
        }
        return ret


ETPol = Polarizability(
    data_source=resource_filename(
        "factorpol", os.path.join("data", "alphas_elements.csv")
    ),
    typing_scheme=PolarizabilityType.Element,
)

SmirnoffPol = Polarizability(
    data_source=resource_filename(
        "factorpol", os.path.join("data", "alphas_smirnoff.csv")
    ),
    typing_scheme=PolarizabilityType.SMIRNOFF,
)


@dataclass
class BondChargeCorrections:
    data_source: str
    polarizability_type: enum.Enum

    @property
    def data(self) -> pd.core.frame.DataFrame:
        dt = pd.read_csv(self.data_source, index_col="BCC SMIRKS")
        return dt

    @property
    def parameters(self) -> Dict:
        if self.polarizability_type == PolarizabilityType.Element:
            column_name = "ET-BCCs (elementary charge)"
        elif self.polarizability_type == PolarizabilityType.SMIRNOFF:
            column_name = "SF-BCCs (elementary charge)"
        else:
            raise NotImplementedError
        ret = {k: v for k, v in zip(self.data.index, self.data[column_name])}
        return ret

    @property
    def recharge_collection(self) -> BCCCollection:
        ret = BCCCollection(
            parameters=[
                BCCParameter(smirks=sm, value=float(vs))
                for sm, vs in self.parameters.items()
            ]
        )
        ret.aromaticity_model = aromaticity_model
        return ret


FactorPolETBccs = BondChargeCorrections(
    data_source=resource_filename("factorpol", os.path.join("data", "bcc_dPol.csv")),
    polarizability_type=PolarizabilityType.Element,
)

FactorPolSFBccs = BondChargeCorrections(
    data_source=resource_filename("factorpol", os.path.join("data", "bcc_dPol.csv")),
    polarizability_type=PolarizabilityType.SMIRNOFF,
)

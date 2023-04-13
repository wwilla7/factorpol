"""
    This module provide functionalities to generate reference QM ESPs and query existing ESPs data

    This module is designed to be used with `Ray` to distribute QM calculations.
"""
import logging
import os
import uuid
from typing import Dict, List

import numpy as np
import ray
from numpy import ndarray
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.esp.storage.db import (DBBase, DBConformerRecord,
                                            DBESPSettings, DBGridSettings,
                                            DBMoleculeRecord, DBPCMSettings)
from openff.recharge.grids import MSKGridSettings
from openff.toolkit.topology import Molecule
from sqlalchemy import select
from sqlalchemy.orm.session import Session

from factorpol.utilities import flatten_a_list

logger = logging.getLogger(__name__)


@ray.remote
def _worker(
    off_mol: Molecule,
    method: str,
    basis: str,
    wd: str,
    thread_per_worker: int,
    n_conf: int = 1,
    msk_density: float = 1.0,
    msk_layers: float = 4.0,
    external_field: ndarray = np.array([0.0, 0.0, 0.0]),
) -> List[MoleculeESPRecord]:
    """
    This function is used to carry out QM ESPs calculations with psi4

    Parameters
    ----------
    off_mol: Molecule
        Input molecule as an OpenFF molecule object

    method: str
        QM method

    basis: str
        QM level of theory

    wd: str
        Working directory

    thread_per_worker: int
        Number of cpus per worker

    n_conf: int
        Number of conformers to generate if the input molecule does not contains coordinates.

    msk_density: float
        Density of MSK grid points

    msk_layers: float
        Number of layer s of MSK grid points

    external_field: np.ndarray
        If impose electric field is desired. Default is none. np.array([0., 0. 0.])

    Returns
    -------
    List[MoleculeESPRecord]
        Returns all QM ESPs records

    """
    qc_data_records = []
    os.makedirs(wd, exist_ok=False)

    qc_data_settings = ESPSettings(
        method=method,
        basis=basis,
        grid_settings=MSKGridSettings(
            type="msk", density=msk_density, layers=msk_layers
        ),
        perturb_dipole=external_field,
    )
    if off_mol.conformers == None:

        conformers = ConformerGenerator.generate(
            off_mol, ConformerSettings(max_conformers=n_conf)
        )

    else:
        conformers = off_mol.conformers

    for idx, conformer in enumerate(conformers):

        conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
            molecule=off_mol,
            conformer=conformer,
            settings=qc_data_settings,
            directory=os.path.join(wd, f"conf{idx:02d}"),
            n_thread=thread_per_worker,
        )
        qc_data_record = MoleculeESPRecord.from_molecule(
            off_mol, conformer, grid, esp, electric_field, qc_data_settings
        )
        qc_data_records.append(qc_data_record)

    return qc_data_records


class QWorker:
    """
    Distribute QM calculations with `Ray`

    Parameters
    ----------
    n_worker: int
        Number of ray workers to run QM calculations

    thread_per_worker: int
        Number of CPUs per ray worker to use
    """

    def __init__(self, n_worker: int = 2, thread_per_worker: int = 8):
        self.n_worker = n_worker
        self.thread_per_worker = thread_per_worker
        self.total_thread = self.n_worker * self.thread_per_worker
        self.records = []
        self.dataset = []
        self.working_directory = None

        ray.shutdown()
        ray.init(num_cpus=self.n_worker, include_dashboard=False)

    def start(
        self,
        dataset: List[Molecule],
        method: str,
        basis: str,
        wd: str,
        n_conf: int = 1,
        msk_density: float = 1.0,
        msk_layers: float = 4.0,
        external_field: ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> List:
        """
        Start a calculation process.

        Parameters
        ----------
        dataset: List[Molecule]
            A list of OpenFF molecules to run QM ESPs calculations

        method: str
            QM method

        basis: str
            QM level of theory

        wd: str
            Working directory

        n_conf: int
            Number of conformers to generate if the input molecule does not contains coordinates.

        msk_density: float
            Density of MSK grid points

        msk_layers: float
            Number of layer s of MSK grid points

        external_field: np.ndarray
            If impose electric field is desired. Default is none. np.array([0., 0. 0.])

        Returns
        -------
        List
            Returns all MoleculeESPRecord in a list

        """
        if os.path.exists(wd):
            subid = str(uuid.uuid4())
            wd = os.path.join(wd, subid)

        self.working_directory = wd
        logger.info(f"Generate QM ESPs with imposed electric field {external_field}.")

        workers = [
            _worker.remote(
                off_mol=mol,
                method=method,
                basis=basis,
                wd=os.path.join(wd, f"mol{idx:02d}"),
                thread_per_worker=self.thread_per_worker,
                n_conf=n_conf,
                msk_density=msk_density,
                msk_layers=msk_layers,
                external_field=external_field,
            )
            for idx, mol in enumerate(dataset)
        ]
        ret = ray.get(workers)
        if len(ret) > 2:
            ret = flatten_a_list(ret)
        self.records.append(ret)
        return ret

    def store(self, my_session: Session, records: List[MoleculeESPRecord]) -> List:
        """
        Store a list of MoleculeESPRecord to the input session associated with user database.

        Parameters
        ----------
        my_session: session.Session
            A working session associated with a database for storing QM ESPs data.

        records:
            A list of MoleculeESPRecords to be stored.

        Returns
        -------
        List
            Returns a list of SMILEs Strings of stored MoleculeESPRecords.

        """

        molecules = [add_molecule(r, my_session) for r in records]
        self.dataset.append(molecules)
        return molecules


def add_molecule(record, my_session):
    """
    Add one record to the input session

    Parameters
    ----------
    record: MoleculeESPRecord
        The record to be stored.

    my_session: session.Session
        A working session associated with a database for storing QM ESPs data.

    Returns
    -------
    str
        Returns the SMILES string of stored record

    """

    tagged_smiles = record.tagged_smiles
    offmol = Molecule.from_mapped_smiles(tagged_smiles)
    smiles = offmol.to_smiles(explicit_hydrogens=False)

    stmt = select(DBMoleculeRecord).where(DBMoleculeRecord.smiles == smiles)
    ret = my_session.scalars(stmt).all()

    if bool(ret):
        table = ret[0]
    else:
        table = DBMoleculeRecord(smiles=smiles)

    grid_settings = MSKGridSettings(
        type="msk",
        density=record.esp_settings.grid_settings.density,
        layers=record.esp_settings.grid_settings.layers,
    )
    esp_settings = ESPSettings(
        method=record.esp_settings.method,
        basis=record.esp_settings.basis,
        grid_settings=grid_settings,
        perturb_dipole=record.esp_settings.perturb_dipole,
    )
    table.conformers.append(
        DBConformerRecord(
            tagged_smiles=record.tagged_smiles,
            coordinates=record.conformer,
            grid=record.grid_coordinates,
            esp=record.esp,
            field=record.electric_field,
            grid_settings=DBGridSettings.unique(my_session, grid_settings),
            pcm_settings=None
            if not record.esp_settings.pcm_settings
            else DBPCMSettings.unique(my_session, record.esp_settings.pcm_settings),
            esp_settings=DBESPSettings.unique(my_session, esp_settings),
        )
    )
    my_session.add(table)
    my_session.commit()
    return smiles


def from_conformer_to_molecule(dbconformer: DBConformerRecord):
    """
    Reconstruct a conformer record to a molecule record.

    Parameters
    ----------
    dbconformer: DBConformerRecord
        A conformer record DB

    Returns
    -------
    MoleculeESPRecord
        Returns a MoleculeESPRecord for this conformation.

    """
    return MoleculeESPRecord(
        tagged_smiles=dbconformer.tagged_smiles,
        conformer=dbconformer.coordinates,
        grid_coordinates=dbconformer.grid,
        esp=dbconformer.esp.reshape([-1, 1]),
        electric_field=dbconformer.field,
        esp_settings=ESPSettings(
            basis=dbconformer.esp_settings.basis,
            method=dbconformer.esp_settings.method,
            grid_settings=DBGridSettings.db_to_instance(dbconformer.grid_settings),
            pcm_settings=None
            if not dbconformer.pcm_settings
            else DBPCMSettings.db_to_instance(dbconformer.pcm_settings),
            perturb_dipole=dbconformer.esp_settings.perturb_dipole,
        ),
    )


def retrieve_by_external_field(
    my_session: Session, molecule: str, eefield: np.ndarray
) -> List:
    """
    Query records according to imposed electric field.

    Parameters
    ----------
    my_session: session.Session
        A working session associated with a database for stored QM ESPs data.

    molecule: str
        The SMILES string of the molecule to query

    eefield: ndarray
        The imposed electric field to filter

    Returns
    -------
    List
        A list of returned records

    """
    db_records = my_session.scalars(
        select(DBConformerRecord)
        .join(DBMoleculeRecord)
        .join(DBESPSettings)
        .where(DBMoleculeRecord.smiles == molecule)
        .where(DBESPSettings.perturb_dipole == eefield)
    ).all()
    return db_records


def retrieve_by_conformation(
    my_session: Session, molecule: str, conformation: np.ndarray
) -> List:
    """
    Filter/retrieve records based on conformation

    Parameters
    ----------
    my_session: session.Session
        A working session associated with a database for stored QM ESPs data.

    molecule: str
        The SMILES string of molecule to filter

    conformation: ndarray
        The conformation coordinates to filter/query

    Returns
    -------
    List
        Returns a list of retrieved records

    """
    db_records = my_session.scalars(
        select(DBConformerRecord)
        .join(DBMoleculeRecord)
        .where(DBMoleculeRecord.smiles == molecule)
        .where(DBConformerRecord.coordinates == conformation)
    ).all()
    return db_records


def rebuild_molecule(my_session: Session, molecule: str) -> Dict:
    """
    Rebuild a molecule with baseline QM ESPs from all conformations.

    Parameters
    ----------
    my_session: session.Session
        A working session associated with a database for stored QM ESPs data.

    molecule: str
        The SMILE String of the molecule of choice.

    Returns
    -------
    Dict
        Returns a dictionary of records
        Conformer names are keys and records are values.

    """
    molecules = retrieve_by_external_field(
        my_session=my_session, molecule=molecule, eefield=np.zeros(3)
    )

    ret = {}
    for idx, conformer in enumerate(molecules):
        ret[f"conf{idx:02d}"] = [
            from_conformer_to_molecule(r)
            for r in retrieve_by_conformation(
                my_session=my_session,
                molecule=molecule,
                conformation=conformer.coordinates,
            )
        ]
    return ret

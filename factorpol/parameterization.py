"""This module is for parameterizing small molecules with typed polarizabilities, AM1-BCC-dPol partial charges,
    and bonded parameters of Open Force Field - Sage or AMBER GAFF"""

import os
import shutil

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

import numpy as np
import parmed as pmd
from lxml import etree
from openff.toolkit.topology import Molecule as off_Molecule
from openff.toolkit.topology import Topology as off_Topology
from openmm.app import ForceField
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)
from parmed.openmm.parameters import OpenMMParameterSet
from openff.units import unit
from factorpol.utilities import PolarizabilityType
from factorpol.bcc_training import BccTrainer
from enum import Enum


def add_force(parent, Name, TypeName, c0, alpha):
    """
    Create a MPID Force for OpenMM to handle induced dipoles.

    :param parent: Parent `ForceField` xml tree
    :param Name: Atom name
    :param TypeName: Atom type
    :param c0: Permanent partial charge
    :param alpha: Typed polarizability parameter
    :return: ForceField xml parent tree
    """

    # Set Thole
    if alpha == 0.0:
        thole = 0.0
    else:
        thole = 8.0

    etree.SubElement(parent, "Multipole", name=Name, type=str(TypeName), c0=str(c0))
    etree.SubElement(
        parent,
        "Polarize",
        name=Name,
        type=str(TypeName),
        polarizabilityXX=str(alpha),
        polarizabilityYY=str(alpha),
        polarizabilityZZ=str(alpha),
        thole=str(thole),
    )

    return parent


def parameterize_molecule(smile: str, ff_name: str, polarizability_type: Enum, output_path: str):
    """
    Parameterize a molecule with force field of choice

    :param smile: SMILES string for molecule to parameterize
    :param ff_name: sage or gaff
    :param output_path: output directory to write ff.xml
    :return: molecular dipole moment of this molecule
    """
    off_mol = off_Molecule.from_smiles(smile)
    off_mol.generate_conformers(n_conformers=1)
    off_topology = off_Topology.from_molecules(off_mol)
    omm_vac_topology = off_topology.to_openmm()
    qcmol = off_mol.to_qcschema()
    symbol = qcmol.symbols

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print("Output path exists. Remove everything.")
    os.makedirs(output_path)

    # Create Openmm system
    forcefield = ForceField()

    if ff_name.lower() in ["gaff"]:
        gaff = GAFFTemplateGenerator(molecules=[off_mol])
        forcefield.registerTemplateGenerator(gaff.generator)
        omm_vac_system = forcefield.createSystem(omm_vac_topology)

    elif ff_name.lower() in ["sage"]:
        offff = SMIRNOFFTemplateGenerator(molecules=[off_mol])
        forcefield.registerTemplateGenerator(offff.generator)
        omm_vac_system = forcefield.createSystem(omm_vac_topology)

    pmd_structure = pmd.openmm.load_topology(omm_vac_topology, system=omm_vac_system)
    charges = []
    alphas = []

    if polarizability_type == PolarizabilityType.Element:
        from factorpol.utilities import ETPol as PolLibrary
        from factorpol.utilities import FactorPolETBccs as BCCdPolLibrary
        patterns = symbol.copy()

    elif polarizability_type == PolarizabilityType.SMIRNOFF:
        from factorpol.utilities import SmirnoffPol as PolLibrary
        from factorpol.utilities import FactorPolSFBccs as BCCdPolLibrary
        from factorpol.utilities import smirnoff_labels, ff
        patterns = smirnoff_labels(off_mol, ff)

    recharge_collection = BCCdPolLibrary.recharge_collection

    for idx, at in enumerate(pmd_structure.atoms):
        if idx < off_mol.n_atoms:
            alphas.append(PolLibrary.parameters[patterns[idx]].to('nm**3').magnitude)
        else:
            pass
        charges.append(at.charge)
        at.charge = 0.0

    np.savetxt(os.path.join(output_path, "am1bcc.dat"), charges, fmt="%10.7f")

    charges = np.round(
        BccTrainer.generate_charges(smile, recharge_collection).reshape(-1), 7
    )
    np.savetxt(os.path.join(output_path, "am1bccdPol.dat"), charges, fmt="%10.7f")

    openmm_params = OpenMMParameterSet()
    openmm_xml = openmm_params.from_structure(pmd_structure)
    get_residues = pmd.modeller.ResidueTemplateContainer.from_structure(
        pmd_structure
    ).to_library()
    openmm_xml.residues.update(get_residues)
    openmm_xml.write(os.path.join(output_path, "forcefield.xml"))

    with open(os.path.join(output_path, "forcefield.xml"), "r") as f:
        ff_text = f.read()

    root = etree.fromstring(ff_text)

    sh = root.find("NonbondedForce")

    sh.set("coulomb14scale", "0.83333")
    sh.set("lj14scale", "0.5")

    mpidforce = etree.SubElement(root, "MPIDForce")
    mpidforce.set("coulomb14scale", "1.0")

    # TypeName, charge, polarizability, thole factor
    # Unit: N/A, e, nanometer**3, thole factor is unitless
    for idx, at in enumerate(pmd_structure.atoms):
        if idx < off_mol.n_atoms:
            mpidforce = add_force(
                mpidforce,
                Name=at.name,
                TypeName=at.type,
                c0=charges[idx],
                alpha=alphas[idx],
            )

    # organize XML file
    tree = etree.ElementTree(root)
    xml = etree.tostring(tree, encoding="utf-8", pretty_print=True).decode("utf-8")
    xml = xml.replace("><", ">\n\t<")
    xml = xml.replace("<MPIDForce", "  <MPIDForce")
    xml = xml.replace("\t</ForceField>", "</ForceField>")
    with open(os.path.join(output_path, "forcefield.xml"), "w") as f:
        f.write(xml)

    off_mol.partial_charges = charges * unit.elementary_charge
    off_mol.to_file(os.path.join(output_path, "molecule.mol2"), file_format="mol2")
    off_mol.to_file(os.path.join(output_path, "molecule.pdb"), file_format="pdb")

    mu = _calculate_dipoles(off_mol)

    return mu


def _calculate_dipoles(offmol: off_Molecule):
    """
    Calculate molecular dipole moment for a parameterized molecule (with partial charges)
    :param offmol: OpenFF Molecule Object
    :return: molecular dipole moment
    """
    charge = offmol.partial_charges.m_as("elementary_charge")
    geometry = offmol.conformers[0].m_as("angstrom")
    dipole = Q_(
        np.linalg.norm(np.sum(np.multiply(charge.reshape(-1, 1), geometry), axis=0)),
        "e*angstrom",
    ).to("debye")
    return dipole

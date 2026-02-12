from pathlib import Path
import openfe
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol, AbsoluteSolvationSolventUnit
from openmm import XmlSerializer
from openff.units.openmm import from_openmm
from openff.units import unit
import numpy as np
import bz2


def serialize_system(system, filename: Path):
    """
    Serialize an OpenMM System.

    Parameters
    ----------
    system : System
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    with bz2.open(filename, mode="wb") as outfile:
        serialized_thing = XmlSerializer.serialize(system)
        outfile.write(serialized_thing.encode())


def serialize_positions(positions, filename: Path):
    """
    Write out a numpy npz file for the positions.
    """
    off_pos = from_openmm(positions)
    pos_arr = off_pos.to('nanometer').m
    np.savez(filename, positions=pos_arr)


mapping = openfe.LigandAtomMapping.from_json('../../../data/ross_2023/jacs/tyk2/tyk2_edge.json')
protein = openfe.ProteinComponent.from_pdb_file('../../../data/ross_2023/jacs/tyk2/protein.pdb')
solvent = openfe.SolventComponent()

# Hybrid Topology

solv_sysA = openfe.ChemicalSystem({'ligand': mapping.componentA, 'solvent': solvent})
solv_sysB = openfe.ChemicalSystem({'ligand': mapping.componentB, 'solvent': solvent})

settings = RelativeHybridTopologyProtocol.default_settings()
settings.solvation_settings.box_shape = 'dodecahedron'
settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
protocol = RelativeHybridTopologyProtocol(settings=settings)

dag = protocol.create(stateA=solv_sysA, stateB=solv_sysB, mapping=mapping)
dag_unit = list(dag.protocol_units)[0]
debug = dag_unit.run(dry=True)['debug']

htf = debug['sampler']._factory

serialize_system(htf.hybrid_system, Path('hybrid_system.xml'))
serialize_positions(htf.hybrid_positions, Path('hybrid_positions.npz'))
serialize_system(htf._old_system, Path('standard_system.xml'))
serialize_positions(htf._old_positions, Path('standard_positions.npz'))

# AHFE

solv_sysB = openfe.ChemicalSystem({'solvent': solvent})
settings = AbsoluteSolvationProtocol.default_settings()
settings.solvation_settings.box_shape = 'dodecahedron'
settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
settings.solvent_forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
protocol = AbsoluteSolvationProtocol(settings=settings)

dag = protocol.create(stateA=solv_sysA, stateB=solv_sysB, mapping=None)
prot_units = list(dag.protocol_units)
sol_unit = [u for u in prot_units if isinstance(u, AbsoluteSolvationSolventUnit)][0]
debug = sol_unit.run(dry=True)["debug"]
serialize_system(debug["alchem_system"], Path("absolute_system.xml"))
serialize_positions(debug["positions"], Path("absolute_positions.npz"))


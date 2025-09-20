from pathlib import Path
import openfe
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
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


mapping = openfe.LigandAtomMapping.from_json('../../data/ross_2023/jacs/tyk2/tyk2_edge.json')
protein = openfe.ProteinComponent.from_pdb_file('../../data/ross_2023/jacs/tyk2/protein.pdb')
solvent = openfe.SolventComponent()

sysA = openfe.ChemicalSystem({'ligand': mapping.componentA, 'protein': protein, 'solvent': solvent})
sysB = openfe.ChemicalSystem({'ligand': mapping.componentB, 'protein': protein, 'solvent': solvent})

settings = RelativeHybridTopologyProtocol.default_settings()
settings.solvation_settings.box_shape = 'dodecahedron'
settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
protocol = RelativeHybridTopologyProtocol(settings=settings)

dag = protocol.create(stateA=sysA, stateB=sysB, mapping=mapping)
dag_unit = list(dag.protocol_units)[0]
debug = dag_unit.run(dry=True)['debug']

htf = debug['sampler']._factory

serialize_system(htf.hybrid_system, Path('hybrid_system.xml'))
serialize_positions(htf.hybrid_positions, Path('hybrid_positions.npz'))
serialize_system(htf._old_system, Path('standard_system.xml'))
serialize_positions(htf._old_positions, Path('standard_positions.npz'))

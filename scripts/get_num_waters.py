import string
import click
import tempfile
import pathlib
import pandas as pd
import gufe
import json
from openff.units import unit
import openfe
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol
from rdkit import Chem
import MDAnalysis as mda


def get_settings():
    """
    Utility method for getting MDProtocol settings.

    These settings mostly follow defaults but use very short
    simulation times to avoid being too much of a burden on users' machines.
    """
    settings = openfe.protocols.openmm_md.plain_md_methods.PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = 1 * unit.picosecond
    settings.simulation_settings.equilibration_length = 1 * unit.picosecond
    settings.simulation_settings.production_length = 1 * unit.picosecond
    settings.solvation_settings.box_shape = 'dodecahedron'
    settings.output_settings.checkpoint_interval = 100 * unit.picosecond
    settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.engine_settings.compute_platform = 'cuda'
    return settings


def get_waters(dagres, protocol):
    """
    Get the number of waters & atoms

    Parameters
    ----------
    dagres : openfe.ProtocolDAGResult
      The Protocol DAG result.
    protocol : openfe.Protocol
      The Protocol we ran.
    """
    protocol_results = protocol.gather([dagres])
    # hack to get the file path
    pdb_filename = protocol_results.get_pdb_filename()[0]
    u = mda.Universe(pdb_filename)
    waters = u.select_atoms("resname HOH NA CL")
    return len(waters.residues), len(u.atoms)


def run_md(dag, protocol):
    """
    Run a DAG and check it was ok.

    Parameters
    ----------
    dag : openfe.ProtocolDAG
      A ProtocolDAG to execute.
    protocol : openfe.Protocol
      The Protocol we are running.

    Raises
    ------
    AssertionError
      If any of the simulation Units failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        dagres = gufe.protocols.execute_DAG(
            dag,
            shared_basedir=workdir,
            scratch_basedir=workdir,
            keep_shared=True,
            raise_error=True,
            n_retries=0,
        )

        if not dagres.ok():
            return 'NaN'
        else:
            return get_waters(dagres, protocol)


def run_inputs(pdb, cofactors, edge):
    """
    Validate input files by running a short MD simulation

    Parameters
    ----------
    pdb : pathlib.Path
      A Path to a protein PDB file.
    cofactors : Optional[pathlib.Path]
      A Path to an SDF file containing the system's cofactors.
    edge : Optional[pathlib.Path]
      A Path to a JSON serialized AtomMapping. ComponentA will
      be used as part of the simulation.
    """
    # Create the solvent and protein components
    solv = openfe.SolventComponent()
    prot = openfe.ProteinComponent.from_pdb_file(str(pdb))

    results = {'complex': 'NaN', 'solvent': 'NaN'}

    for leg in results.keys():
        components_dict = {
            'solvent': solv,
        }

        if leg == 'complex':
            components_dict['protein'] = prot

            # If we have cofactors, populate them and store them based on
            # an single letter index (we assume no more than len(alphabet) cofactors)
            if cofactors is not None:
                cofactors = [
                    openfe.SmallMoleculeComponent(m)
                    for m in Chem.SDMolSupplier(str(cofactors), removeHs=False)
                ]
        
                for cofactor, entry in zip(cofactors, string.ascii_lowercase):
                    components_dict[entry] = cofactor

        if edge is not None:
            mapping = openfe.LigandAtomMapping.from_json(edge)
            components_dict['ligand'] = mapping.componentA

        # Create the ChemicalSystem
        system = openfe.ChemicalSystem(components_dict)
    
        # Get the settings and create the protocol
        settings = get_settings()

        if leg == 'solvent':
            settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer

        protocol = PlainMDProtocol(settings=settings)

        # Now create the DAG and run it
        dag = protocol.create(stateA=system, stateB=system, mapping=None)

        results[leg] = run_md(dag, protocol)
        
    return results


@click.command
@click.option(
    '--protein',
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    required=True,
    default="protein.pdb",
    help="Path to the protein PDB",
)
@click.option(
    '--edge',
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    default="edge.json",
    help="Path to the ligand transform edge",
)
@click.option(
    '--cofactors',
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    default=None,
    help="Path to the cofactors",
)
def run(protein, edge, cofactors):
    """
    Run things.
    """
    results = run_inputs(pdb=protein, cofactors=cofactors, edge=edge)
    print(results)


if __name__ == "__main__":
    run()

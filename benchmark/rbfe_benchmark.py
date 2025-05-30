import string
import click
import tempfile
import pathlib
import pandas as pd
import gufe
import json
import yaml
from openff.units import unit
import openfe
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings
from rdkit import Chem


def get_settings(waters):
    """
    Utility method for getting Protocol settings.
    """
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.simulation_settings.equilibration_length = 100 * unit.picosecond
    settings.simulation_settings.production_length = 500 * unit.picosecond
    settings.simulation_settings.time_per_iteration = 2.5 * unit.picosecond
    settings.simulation_settings.real_time_analysis_interval = 100 * unit.picosecond
    settings.output_settings.checkpoint_interval = 100 * unit.picosecond
    settings.output_settings.positions_write_frequency = 100 * unit.picosecond
    settings.solvation_settings = OpenMMSolvationSettings(
        number_of_solvent_molecules=waters,
        box_shape='dodecahedron',
        solvent_padding=None,
    )
    settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = "cuda"
    return settings


def get_performance(dagres, protocol):
    """
    Get the final ns/day performance

    Parameters
    ----------
    dagres : openfe.ProtocolDAGResult
      The Protocol DAG result.
    protocol : openfe.Protocol
      The Protocol we ran.
    """
    protocol_results = protocol.gather([dagres])
    # hack to get the file path
    nc = [purs[0].outputs["nc"] for purs in protocol_results.data.values()][0]
    filepath = nc.resolve().parent
    log = filepath / "simulation_real_time_analysis.yaml"
    with open(log) as stream:
        data = yaml.safe_load(stream)

    return data[-1]["timing_data"]["ns_per_day"]


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
            return "NaN"
        else:
            val = get_performance(dagres, protocol)
            print(val)
            return val


def run_inputs(pdb, cofactors, edge, waters):
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
    waters : dict[str, int]
      A dictionary keyed by the legs of the simulation with
      the number of waters to run.
    """
    # Create the solvent and protein components
    solv = openfe.SolventComponent()
    prot = openfe.ProteinComponent.from_pdb_file(str(pdb))

    results = {'solvent': "NaN", 'complex': "NaN"}

    for leg in results.keys():
        # Store there in a components dictionary
        stateA_dict = {
            "solvent": solv,
        }

        stateB_dict = {
            "solvent": solv,
        }

        if leg == "complex":
            stateA_dict["protein"] = prot
            stateB_dict["protein"] = prot

            # If we have cofactors, populate them and store them based on
            # an single letter index (we assume no more than len(alphabet) cofactors)
            if cofactors is not None:
                cofactors = [
                    openfe.SmallMoleculeComponent(m)
                    for m in Chem.SDMolSupplier(str(cofactors), removeHs=False)
                ]
    
                for cofactor, entry in zip(cofactors, string.ascii_lowercase):
                    stateA_dict[entry] = cofactor
                    stateB_dict[entry] = cofactor

        if edge is not None:
            mapping = openfe.LigandAtomMapping.from_json(edge)
            stateA_dict["ligand"] = mapping.componentA
            stateB_dict["ligand"] = mapping.componentB

        # Create the ChemicalSystem
        stateA = openfe.ChemicalSystem(stateA_dict)
        stateB = openfe.ChemicalSystem(stateB_dict)

        # Get the settings and create the protocol
        settings = get_settings(waters[leg])
        protocol = RelativeHybridTopologyProtocol(settings=settings)

        # Now create the DAG and run it
        dag = protocol.create(stateA=stateA, stateB=stateB, mapping=mapping)

        results[leg] = run_md(dag, protocol)

    return results


@click.command
@click.option(
    "--input_file",
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    required=True,
    help="Path to the benchmark input file",
)
@click.option(
    "--output_file",
    type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path),
    default="rbfe_benchmark.out",
    help="Path to the benchmark output file",
)
def run_benchmark(input_file, output_file):
    """
    Run a benchmark.
    """
    data_path = input_file.resolve().parent

    with open(input_file, "r") as f:
        benchmark = json.loads(f.read())

    benchmark_results = {}

    for system in benchmark:
        pdb = data_path / benchmark[system]["protein"]
        edge = data_path / benchmark[system]["edge"]
        if "cofactors" in benchmark[system]:
            cofactors = data_path / benchmark[system]["cofactors"]
        else:
            cofactors = None
        waters = benchmark[system]["waters"]
        benchmark_results[system] = run_inputs(pdb=pdb, cofactors=cofactors, edge=edge, waters=waters)

    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)


if __name__ == "__main__":
    run_benchmark()

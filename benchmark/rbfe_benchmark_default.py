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
    settings.simulation_settings.real_time_analysis_interval = 100 * unit.picosecond
    settings.output_settings.checkpoint_interval = 100 * unit.picosecond
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = "cuda"
    settings.alchemical_settings.explicit_charge_correction = True
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

    def iter_path_like_values(value):
        if isinstance(value, (str, pathlib.Path)):
            yield pathlib.Path(value)
            return
        if hasattr(value, "resolve"):
            try:
                yield pathlib.Path(value)
            except TypeError:
                pass
            return
        if isinstance(value, dict):
            for nested_value in value.values():
                yield from iter_path_like_values(nested_value)
            return
        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                yield from iter_path_like_values(nested_value)

    def extract_ns_per_day(log):
        if not log.exists():
            return None
        with open(log) as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, list) or not data:
            return None
        timing_data = data[-1].get("timing_data")
        if not isinstance(timing_data, dict):
            return None
        ns_per_day = timing_data.get("ns_per_day")
        if ns_per_day is None:
            return None
        return ns_per_day

    nc = None
    observed_output_keys = set()
    path_candidates = []
    for protocol_unit_results in protocol_results.data.values():
        for unit_result in protocol_unit_results:
            outputs = getattr(unit_result, "outputs", {})
            if not isinstance(outputs, dict):
                continue
            observed_output_keys.update(str(key) for key in outputs.keys())
            for output_value in outputs.values():
                path_candidates.extend(iter_path_like_values(output_value))
            candidate = outputs.get("nc")
            if candidate is not None:
                nc = candidate
                break
        if nc is not None:
            break

    if nc is not None:
        filepath = (
            nc.resolve().parent
            if hasattr(nc, "resolve")
            else pathlib.Path(nc).parent
        )
        ns_per_day = extract_ns_per_day(filepath / "simulation_real_time_analysis.yaml")
        if ns_per_day is not None:
            return ns_per_day

    log_filenames = (
        "simulation_real_time_analysis.yaml",
        "real_time_analysis.yaml",
    )
    for path_candidate in path_candidates:
        candidate_dirs = []
        if path_candidate.is_dir():
            candidate_dirs.append(path_candidate)
        candidate_dirs.append(path_candidate.parent)
        candidate_dirs.append(path_candidate.parent.parent)
        for candidate_dir in candidate_dirs:
            for log_filename in log_filenames:
                ns_per_day = extract_ns_per_day(candidate_dir / log_filename)
                if ns_per_day is not None:
                    return ns_per_day

    observed = ", ".join(sorted(observed_output_keys)) or "<none>"
    raise KeyError(
        "Unable to resolve benchmark performance from gathered protocol unit "
        f"outputs. Observed output keys: {observed}."
    )


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
            raise_error=False,
            n_retries=3,
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
            try:
                mapping = openfe.LigandAtomMapping.from_json(edge)
            except AttributeError:
                with open(edge, 'r') as fd:
                    mapping = openfe.LigandAtomMapping.from_dict(json.load(fd))

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

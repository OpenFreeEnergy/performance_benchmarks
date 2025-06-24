import openmm
from openmm import Platform
from openmm import unit
from openff.units import unit as offunit
from openff.units.openmm import to_openmm
from openmm import XmlSerializer
import numpy as np
import bz2
import time


CUDA_PLATFORM = Platform.getPlatformByName('CUDA')


def benchmark(
    system,
    positions,
    nsteps=100000,
    timestep=1.0*unit.femtoseconds,
    platform=CUDA_PLATFORM,
    tag="alchemical",
):
    """
    Benchmark the performance of a system.
    """
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    
    print(f"running {tag} system")
    t0 = time.time()
    integrator.step(nsteps)
    t1 = time.time()
    print(f"finished running {tag} system")
    return t1 - t0


def deserialize(xml, npz):
    """
    Deserialize an OpenMM system in xml format and positions stored in npz.
    """
    with bz2.open(xml, 'rb') as file:
        thing = file.read().decode()
        system = XmlSerializer.deserialize(thing)

    off_positions = np.load(npz)['positions'] * offunit.nanometer
    return system, to_openmm(off_positions)


results = {}

for system_type in ['hybrid', 'standard']:
    system, positions = deserialize(
        f"{system_type}_system.xml",
        f"{system_type}_positions.npz",
    )

    results[system_type] = benchmark(system, positions, tag=system_type)
    print(f"{system_type} simulation time: ", results[system_type])


speedup = results['standard'] / results['hybrid']
print("Speedup: ", speedup)

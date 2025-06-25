import openmm
from openmm import Platform
from openmm import unit
from openff.units import unit as offunit
from openff.units.openmm import to_openmm
from openmm import XmlSerializer
import openmmtools
from openfe.protocols.openmm_rfe._rfe_utils import lambdaprotocol
from openfe.protocols.openmm_rfe._rfe_utils.multistate import HybridRepexSampler
import numpy as np
import bz2
import time


CUDA_PLATFORM = Platform.getPlatformByName('CUDA')


class DummyFactory:
    def __init__(self, system, positions):
        self.hybrid_system = system
        self.hybrid_positions = positions


def adjust_system(
    system
):
    """
    Adjust the OpenMM system properties.
    """
    # Set Ewald tolerance
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setEwaldErrorTolerance(1e-5)


def benchmark_md(
    system,
    positions,
    nsteps=1200000,
    timestep=4.0*unit.femtoseconds,
    platform=CUDA_PLATFORM,
    tag="alchemical",
):
    """
    Benchmark the performance of a system for a conventional
    MD simulation using LangevinMiddleIntegrator.
    """
    integrator = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin,
        1.0 / unit.picosecond,
        timestep
    )
    integrator.setConstraintTolerance(1e-6)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    openmm.LocalEnergyMinimizer.minimize(context)
    
    print(f"running {tag} system")
    t0 = time.time()
    integrator.step(nsteps)
    t1 = time.time()
    print(f"finished running {tag} system")
    return t1 - t0


def benchmark_multistate(
    system,
    positions,
    nsteps=1200000,
    timestep=4.0*unit.femtoseconds,
    windows=12,
    steps_per_exchange=625,
    checkpoint_interval=10000,
    position_interval=0,
    velocity_interval=0,
    online_analysis_interval=40,
    online_analysis_minimum_iterations=0,
    platform=CUDA_PLATFORM,
):
    lambdas = lambdaprotocol.LambdaProtocol(
        functions='default',
        windows=windows,
    )

    reporter = openmmtools.multistate.MultiStateReporter(
        storage='simulation.nc',
        checkpoint_interval=checkpoint_interval,
        checkpoint_storage='checkpoint.nc',
        position_interval=position_interval,
        velocity_interval=velocity_interval,
    )

    integrator = openmmtools.mcmc.LangevinDynamicsMove(
        timestep=timestep,
        collision_rate=1.0 / unit.picosecond,
        n_steps=steps_per_exchange,
        reassign_velocities=False,
        constraint_tolerance=1e-6,
    )

    sampler = HybridRepexSampler(
        mcmc_moves=integrator,
        hybrid_factory=DummyFactory(system, positions),
        online_analysis_interval=online_analysis_interval,
        online_analysis_minimum_iterations=online_analysis_minimum_iterations,
    )

    sampler.setup(
        n_replicas=windows,
        reporter=reporter,
        lambda_protocol=lambdas,
        temperature=298.15 * unit.kelvin,
        endstates=False,
        minimization_platform=platform.getName()
    )

    sampler.energy_context_cache = openmmtools.cache.ContextCache(
        capacity=None,
        time_to_live=None,
        platform=platform,
    )

    sampler.sampler_context_cache = openmmtools.cache.ContextCache(
        capacity=None,
        time_to_live=None,
        platform=platform,
    )

    sampler.minimize(max_iterations=1000)
    t0 = time.time()
    sampler.extend(int(int(nsteps/windows)/steps_per_exchange))
    t1 = time.time()

    return t1-t0


def deserialize(xml, npz):
    """
    Deserialize an OpenMM system in xml format and positions stored in npz.
    """
    with bz2.open(xml, 'rb') as file:
        thing = file.read().decode()
        system = XmlSerializer.deserialize(thing)

    off_positions = np.load(npz)['positions'] * offunit.nanometer
    return system, to_openmm(off_positions)


md_results = {}
multi_results = {}

for system_type in ['hybrid', 'standard']:
    system, positions = deserialize(
        f"{system_type}_system.xml",
        f"{system_type}_positions.npz",
    )

    adjust_system(system)

    md_results[system_type] = benchmark_md(system, positions, tag=system_type)
    print(f"{system_type} MD simulation time: ", md_results[system_type])

    if system_type == 'hybrid':
        multi_results[system_type] = benchmark_multistate(system, positions)


system_speedup = md_results['standard'] / md_results['hybrid']
print("Hybrid system speedup: ", system_speedup)
sampler_speedup = md_results['hybrid'] / multi_results['hybrid']
print("Sampler speedup: ", sampler_speedup)

import openmm
from openmm import Platform, OpenMMException
from openmm import unit
from openff.units import unit as offunit
from openff.units.openmm import to_openmm
from openmm import XmlSerializer
import openmmtools
from openmmtools.alchemy import AlchemicalState
from openmmtools.states import (
    GlobalParameterState,
    SamplerState,
    ThermodynamicState,
    create_thermodynamic_state_protocol,
)
from openfe.protocols.openmm_rfe._rfe_utils import lambdaprotocol
from openfe.protocols.openmm_rfe._rfe_utils.multistate import HybridRepexSampler
import numpy as np
import bz2
import time


CUDA_PLATFORM = Platform.getPlatformByName("CUDA")


class DummyFactory:
    def __init__(self, system, positions):
        self.hybrid_system = system
        self.hybrid_positions = positions


def adjust_system(system):
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
    nsteps=2400000,
    timestep=4.0 * unit.femtoseconds,
    platform=CUDA_PLATFORM,
    tag="alchemical",
):
    """
    Benchmark the performance of a system for a conventional
    MD simulation using LangevinMiddleIntegrator.
    """
    integrator = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, timestep
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


def benchmark_hybrid_multistate(
    system,
    positions,
    nsteps=2800000,
    timestep=4.0 * unit.femtoseconds,
    windows=14,
    steps_per_exchange=625,
    checkpoint_interval=10000,
    position_interval=0,
    velocity_interval=0,
    online_analysis_interval=320,
    online_analysis_minimum_iterations=0,
    platform=CUDA_PLATFORM,
):
    lambdas = lambdaprotocol.LambdaProtocol(
        functions="default",
        windows=windows,
    )

    reporter = openmmtools.multistate.MultiStateReporter(
        storage="simulation.nc",
        checkpoint_interval=checkpoint_interval,
        checkpoint_storage="checkpoint.nc",
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
        minimization_platform=platform.getName(),
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
    sampler.extend(int(int(nsteps / windows) / steps_per_exchange))
    t1 = time.time()

    return t1 - t0


def benchmark_absolute_multistate(
    system,
    positions,
    nsteps=2800000,
    timestep=4.0 * unit.femtoseconds,
    steps_per_exchange=1250,
    checkpoint_interval=10000,
    position_interval=0,
    velocity_interval=0,
    online_analysis_interval=320,
    online_analysis_minimum_iterations=0,
    platform=CUDA_PLATFORM,
):
    lambda_elec=[
        0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    lambda_vdw=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.24,
        0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0
    ]
    lambdas = dict()
    lambdas["lambda_electrostatics"] = [1-x for x in lambda_elec]
    lambdas["lambda_sterics"] = [1-x for x in lambda_vdw]
    alchemical_state = AlchemicalState.from_system(system)
    constants = dict()
    constants["temperature"] = to_openmm(298.15 * offunit.kelvin)
    constants["pressure"] = to_openmm(1 * offunit.bar)
    cmp_states = create_thermodynamic_state_protocol(
        system,
        protocol=lambdas,
        constants=constants,
        composable_states=[alchemical_state]
    )
    sampler_state = SamplerState(positions=positions)
    sampler_state.box_vectors = system.getDefaultPeriodicBoxVectors()
    sampler_states = [sampler_state for _ in cmp_states]

    reporter = openmmtools.multistate.MultiStateReporter(
        storage="simulation.nc",
        checkpoint_interval=checkpoint_interval,
        checkpoint_storage="checkpoint.nc",
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

    sampler = openmmtools.multistate.ReplicaExchangeSampler(
        mcmc_moves=integrator,
        online_analysis_interval=online_analysis_interval,
        online_analysis_minimum_iterations=online_analysis_minimum_iterations,
    )

    sampler.create(thermodynamic_states=cmp_states, sampler_states=sampler_states, storage=reporter)

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
    windows=14
    sampler.extend(int(int(nsteps / windows) / steps_per_exchange))
    t1 = time.time()

    return t1 - t0


def deserialize(xml, npz):
    """
    Deserialize an OpenMM system in xml format and positions stored in npz.
    """
    with bz2.open(xml, "rb") as file:
        thing = file.read().decode()
        system = XmlSerializer.deserialize(thing)

    off_positions = np.load(npz)["positions"] * offunit.nanometer
    return system, to_openmm(off_positions)


md_results = {}
for system_type in ["absolute", "hybrid", "standard"]:
    system, positions = deserialize(
        f"{system_type}_system.xml",
        f"{system_type}_positions.npz",
    )

    adjust_system(system)

    #md_results[system_type] = benchmark_md(system, positions, tag=system_type)
    #print(f"{system_type} MD simulation time: ", md_results[system_type])
    # if system_type == "hybrid":
    #     results = benchmark_hybrid_multistate(system, positions)
    #     print("hybrid multistate:", results)
    if system_type == "absolute":
        results = benchmark_absolute_multistate(system, positions)
        print("absolute multistate: ", results)

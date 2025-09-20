import openmm
from openmm import Platform, OpenMMException
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


def benchmark_multistate(
    system,
    positions,
    nsteps=2400000,
    timestep=4.0 * unit.femtoseconds,
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


def _get_reduced_potentials(context, states, index, temperature=298.15 * unit.kelvin, pressure=1 * unit.bar):

    reduced_potentials = np.zeros(len(states))

    # some constants
    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)

    # Loop through all states and update positions
    for i, state in enumerate(states):
        positions = state.getPositions()
        box_vecs = state.getPeriodicBoxVectors()
        context.setPositions(positions)
        context.setPeriodicBoxVectors(*box_vecs)
        new_state = context.getState(energy=True)

        potential = new_state.getPotentialEnergy() / unit.AVOGADRO_CONSTANT_NA

        if pressure is not None:
            potential += pressure * new_state.getPeriodicBoxVolume()

        reduced_potentials[i] = potential * beta

    # Be careful and update the positions back to their initial values
    init_pos = states[index].getPositions()
    init_vecs = states[index].getPeriodicBoxVectors()
    context.setPositions(init_pos)
    context.setPeriodicBoxVectors(*box_vecs)

    return reduced_potentials


def _get_reduced_potentials2(context, lambda_schedule, index, temperature=298.15 * unit.kelvin, pressure=1 * unit.bar):

    reduced_potentials = np.zeros(len(lambda_schedule.lambda_schedule))

    # some constants
    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)

    # Loop through all lambda states and update parameters
    for i, l in enumerate(lambda_schedule.lambda_schedule):
        for key in lambda_schedule.functions:
            value = lambda_schedule.functions[key](l)
            context.setParameter(key, value)

        state = context.getState(energy=True)
        potential = state.getPotentialEnergy() / unit.AVOGADRO_CONSTANT_NA

        if pressure is not None:
            potential += pressure * state.getPeriodicBoxVolume()

        reduced_potentials[i] = potential * beta

    # Set context parameters back to their initial values
    l = lambda_schedule.lambda_schedule[index]
    for key in lambda_schedule.functions:
        value = lambda_schedule.functions[key](l)
        context.setParameter(key, value)

    return reduced_potentials


def benchmark_manual_multistate(
    system,
    positions,
    nsteps=2400000,
    timestep=4.0 * unit.femtoseconds,
    windows=12,
    steps_per_exchange=625,
    checkpoint_interval=10000,
    position_interval=0,
    velocity_interval=0,
    online_analysis_interval=40,
    online_analysis_minimum_iterations=0,
    platform=CUDA_PLATFORM,
    max_retries=5,
):
    lambda_schedule = lambdaprotocol.LambdaProtocol(
        functions="default",
        windows=windows,
    )

    states = []
    contexts = []
    integrators = []

    print("manual multistate: minimizing")
    for i in range(windows):
        print(f"minimizing: {i}")
        integrator = openmm.LangevinMiddleIntegrator(
            298.15 * unit.kelvin, 1.0 / unit.picosecond, timestep
        )
        integrator.setConstraintTolerance(1e-6)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        lambda_value = lambda_schedule.lambda_schedule[i]

        for key in lambda_schedule.functions:
            value = lambda_schedule.functions[key](lambda_value)
            context.setParameter(key, value)

        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=1000)

        context.setVelocitiesToTemperature(298.15)

        new_state = context.getState(getPositions=True, getVelocities=False)
        states.append(new_state)
        contexts.append(context)
        integrators.append(integrator)

    t0 = time.time()

    n_iterations = int(int(nsteps / windows) / steps_per_exchange)
    reduced_potentials = np.zeros((windows, n_iterations, windows))
    print(f"Number of iterations: {n_iterations}")

    for it in range(n_iterations):
        if it % 10 == 0:
            print(f"iteration: {it}")

        for replica in range(windows):
            # Set state again to replicate the cost of assigning the state back
            contexts[replica].setState(states[replica])

            for attempt in range(max_retries):
                try:
                    integrators[replica].step(steps_per_exchange)
                    new_state = contexts[replica].getState(
                        getPositions=True, getVelocities=True
                    )
                    states[replica] = new_state
                except OpenMMException:
                    if attempt == max_retries - 1:
                        raise ValueError(f"NaNed at iteration {it} replica {replica}")
                    else:
                        print("NaN detected, randomizing velocities and trying again")
                        contexts[replica].setState(states[replica])
                        contexts[replica].setVelocitiesToTemperature(298.15)
                    continue
                else:
                    break
            # First approach is too slow because it uploads coordinates to the device too often
            # reduced_potentials[replica][it] = _get_reduced_potentials(contexts[replica], states, replica)
            reduced_potentials[replica][it] = _get_reduced_potentials2(contexts[replica], lambda_schedule, replica)

        # TODO: add a call here where we swap the state indices to do the exchange
        # This *shouldn't* add much overhead, because all we're doing is using energies already in the reduced potentials
        # and swapping the state indices.

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
multi_results = {}
multi_results_manual = {}

for system_type in ["hybrid", "standard"]:
    system, positions = deserialize(
        f"{system_type}_system.xml",
        f"{system_type}_positions.npz",
    )

    adjust_system(system)

    md_results[system_type] = benchmark_md(system, positions, tag=system_type)
    print(f"{system_type} MD simulation time: ", md_results[system_type])

    if system_type == "hybrid":
        multi_results[system_type] = benchmark_multistate(system, positions)
        print(multi_results[system_type])
        multi_results_manual[system_type] = benchmark_manual_multistate(
            system, positions
        )
        print(multi_results_manual[system_type])


system_speedup = md_results["standard"] / md_results["hybrid"]
print("Hybrid system speedup: ", system_speedup)
sampler_speedup = md_results["hybrid"] / multi_results["hybrid"]
print("MD/OpenMMTools HREX speedup: ", sampler_speedup)
manual_sampler_speedup = md_results["hybrid"] / multi_results_manual["hybrid"]
print("MD/Manual HREX speedup: ", manual_sampler_speedup)

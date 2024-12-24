import time
import numpy as np
import datetime

import boundary_conditions
from grids import Grid1D3V
import maxwell
import newton
from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import *
from results import ResultsND
from time_constraint import calculate_dt_max


def simulate(electrons: Particles, ions: Particles, params: Parameters):
    np.random.seed(params.seed)
    t_start = time.time()
    if electrons.N != ions.N:
        print("Plasma with non-neutral charge")

    if params.bc is BoundaryCondition.Absorbing:
        assert params.damping_width > 0, "If using absorbing boundary condition, the damping width must be set"

    grid = Grid1D3V(params.x_max, params.dx)
    results = ResultsND()

    max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
    dt = calculate_dt_max(params.dx, max_v, electrons.qm, electrons.dimX, safety_factor=20)
    # At t = 0 we know x^0, v^0 and take the fields to be zero
    # Calculate densities n_e(x), n_i(x) and rho(x) at t=0
    grid.set_densities(electrons, ions)

    # Initialize v^(n+1/2)
    newton.initialize_velocities_half_step_1D3V(grid, electrons, ions, params, dt)

    # Save data at time = 0
    KE = electrons.kinetic_energy() + ions.kinetic_energy()
    PE = (grid.dx / 2) * (eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0)
    TE = KE + PE
    results.save(0, KE, PE, TE, grid, electrons, ions)

    print(f"Setup phase took {time.time() - t_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    t_start = time.time()
    t_last = t_start

    t = 0  # Time in the simulation
    step = 0  # Number of iterations
    while t < params.t_max and step < params.max_iter:
        step += 1

        max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
        dt = calculate_dt_max(params.dx, max_v, electrons.qm, electrons.dimX, safety_factor=20)
        t += dt

        # Calculate J⁻ = q * v^(n+1/2) * n^(n)
        maxwell.calc_curr_dens_1D3V(grid, electrons, ions)

        # Store J⁻
        J_prev = grid.J.copy()

        # Calculate positions x^(n+1)
        # depending on the boundary condition, the positions have to be updated before or after
        # the boundary conditions have been applied
        if params.bc is BoundaryCondition.Open:
            # newton.advance_positions(electrons, dt)
            # newton.advance_positions(ions, dt)
            newton.advance_jit(electrons.x, electrons.v, dt)
            newton.advance_jit(ions.x, ions.v, dt)
            boundary_conditions.open_bc(electrons, ions, params.x_max, params.dx)

        elif params.bc is BoundaryCondition.Periodic:
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)
            boundary_conditions.periodic_bc(electrons, ions, params.x_max)

        elif params.bc is BoundaryCondition.Absorbing:
            # Absorbing bc's affect the *velocities* of the particles, so advance positions only
            # after damping of velocities has been calculated
            # TODO: 1D -> 1D3V @Simon should already be fine as the only velocity that matters is the velocity in the x direction as the others don't affect the position of the particle in the system
            # some velocity only needs to be absorbed artificially if the particle gets to close to the edges which can only occur in the x direction
            boundary_conditions.absorbing_bc_1D(electrons, ions, params.x_max, params.damping_width)
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)

        # Calculate densities n_e^(n+1), n_i^(n+1) and rho^(n+1)
        grid.set_densities(electrons, ions)

        # Calculate J⁺ = q * v^(n+1/2) * n^(n+1)
        maxwell.calc_curr_dens_1D3V(grid, electrons, ions)

        # Calculate J^(n+1/2) = (J⁺ + J⁻) / 2 = q * v^(n+1/2) * (n^(n+1) + n^(n)) / 2 + O(dt^2)
        grid.J = (grid.J + J_prev) / 2

        # Calculate the fields E^(n+1), B^(n+1)
        maxwell.calc_fields_1D3V(grid, dt, params.bc)

        # Calculate velocities v^(n+3/2) using the boris pusher
        newton.boris_pusher_1D3V(grid, electrons, dt)
        newton.boris_pusher_1D3V(grid, ions, dt)

        # Calculate the kinetic energy at the staggered timestep before the data is lost
        # To avoid making the calculation every timestep check if it is the step before the save step
        if (step + 1) % 50 == 0:
            KE_prev = electrons.kinetic_energy() + ions.kinetic_energy()

        # Save results every 50 iterations
        if step % 50 == 0:
            KE = (electrons.kinetic_energy() + ions.kinetic_energy() + KE_prev) / 2
            PE = (grid.dx / 2) * (eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0)
            TE = KE + PE
            results.save(t, KE, PE, TE, grid, electrons, ions)

        # Log progress every 5 seconds
        if time.time() - t_last > 5:
            t_last = time.time()
            print(f"{step:9}{t:12.4e}{dt:12.4e}{t_last - t_start:21.3e}{TE:14.4e}")

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time() - t_start:21.3e}{TE:14.4e}")
    print("DONE!")
    string_time = datetime.datetime.fromtimestamp(t_start).strftime("%Y-%m-%dT%Hh%Mm%Ss")
    results.write(f"Results/{string_time}")
    print(f"Results saved in Results/{string_time}/")
    return results

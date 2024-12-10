import time
import numpy as np

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

    max_v = max(np.max(np.linalg.norm(electrons.v, axis=1)), np.max(np.linalg.norm(ions.v, axis=1)))
    dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
    # Calculate densities n_e(x), n_i(x) and rho(x)
    grid.set_densities(electrons, ions)
    # Calculate the fields here so that they are stored at t=0
    grid.set_densities(electrons, ions)
    maxwell.calc_curr_dens(grid, electrons, ions)
    maxwell.calc_fields_1D3V(grid, dt, params.bc)

    # Save data at time = 0
    KE = electrons.kinetic_energy() + ions.kinetic_energy()
    PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
    TE = KE + PE
    results.save(0, KE, PE, TE, grid, electrons, ions)

    print(f"Setup phase took {time.time() - t_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    t_start = time.time()
    t_last = t_start

    t = 0  # Time in the simulation
    step = 0  # Number of iterations
    while t < params.t_max:
        step += 1

        max_v = max(np.max(np.linalg.norm(electrons.v, axis=1)), np.max(np.linalg.norm(ions.v, axis=1)))
        dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
        t += dt

        # Calculate densities n_e(x), n_i(x) and rho(x)
        grid.set_densities(electrons, ions)
        # Calculate current density (Jx(x), Jy(x), Jz(x))
        maxwell.calc_curr_dens(grid, electrons, ions)
        # Solve the Maxwell equations:
        # - TODO: Determine Ex via Gauss?
        # - Determine By and Bz via Faraday
        # - Determine Ey and Ez via Ampère
        # - Bx fixed because 1D spatial variation
        maxwell.calc_fields_1D3V(grid, dt, params.bc)

        # Calculate velocities v^(n+1) using the boris pusher
        newton.boris_pusher_1D3V(grid, electrons, dt)
        newton.boris_pusher_1D3V(grid, ions, dt)

        # Calculate positions x^(n+1)
        # depending on the boundary condition, the positions have to be updated before or after
        # the boundary conditions have been applied
        if params.bc is BoundaryCondition.Open:
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)
            boundary_conditions.open_bc(electrons, ions, params.x_max, params.dx)

        elif params.bc is BoundaryCondition.Periodic:
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)
            boundary_conditions.periodic_bc(electrons, ions, params.x_max)

        elif params.bc is BoundaryCondition.Absorbing:
            # Absorbing bc's affect the *velocities* of the particles, so advance positions only
            # after damping of velocities has been calculated
            # TODO: 1D -> 1D3V @Simon should already be fine as the only velocity that matters is the velocity in the x direction as the others don't affect the position of thge particle in the system
            # some velocity only needs to be absorbed artificially if the particle gets to close to the edges which can only occur in the x direction
            boundary_conditions.absorbing_bc_1D(electrons, ions, params.x_max, params.damping_width)
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)

        # Save results every 50 iterations
        if step % 50 == 0:
            KE = electrons.kinetic_energy() + ions.kinetic_energy()
            PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
            TE = KE + PE
            results.save(t, KE, PE, TE, grid, electrons, ions)

        # Log progress every 5 seconds
        if time.time() - t_last > 5:
            t_last = time.time()
            print(f"{step:9}{t:12.4e}{dt:12.4e}{t_last - t_start:21.3e}{TE:14.4e}")

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time() - t_start:21.3e}{TE:14.4e}")
    # TODO: maybe, once we run *A LOT* of iterations, periodically save the data to a file
    # instead of keeping everything in memory
    return results

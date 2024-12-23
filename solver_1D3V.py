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
    dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
    # Calculate densities n_e(x), n_i(x) and rho(x) at t=0
    grid.set_densities(electrons, ions)
    # Calculate the x-component of the electric field at t=0
    # Initialize electric field at t = 0
    # maxwell.euler_solver_1D3V(grid, dt, params.bc)
    maxwell.calc_E_1D3V(grid, -dt / 2, params.bc)

    # Initialize velocities at t = -dt/2
    newton.initialize_velocities_half_step_1D3V(grid, electrons, ions, params, -dt)

    # Calculate J at t = -1/2dt
    maxwell.calc_curr_dens(grid, electrons, ions)
    # Calculate B at t = -1/2dt (will only be non zero if there is an external B-field E-field )
    maxwell.calc_B_1D3V(grid, -dt / 2, params.bc)
    # Calculate the potential energy due to the B-field at t = -1/2dt
    B_pot = np.sum(grid.B**2) / mu_0
    # Calculate E at t = 0 using J and B at t = -dt/2
    maxwell.calc_E_1D3V(grid, dt, params.bc)
    # Calculate B^(1/2) using E^0
    maxwell.calc_B_1D3V(grid, dt, params.bc)
    # Calculate v^(1/2) and J^(1/2)
    newton.boris_pusher_1D3V(grid, electrons, dt)
    newton.boris_pusher_1D3V(grid, ions, dt)
    maxwell.calc_curr_dens(grid, electrons, ions)

    # Save data at time = 0
    KE = electrons.kinetic_energy() + ions.kinetic_energy()
    PE = (grid.dx / 2) * (eps_0 * np.sum(grid.E**2) + (B_pot + np.sum(grid.B**2) / mu_0))  # average between B(t=-1/2dt) and B(t=1/2dt)
    TE = KE + PE
    results.save(0, KE, PE, TE, grid, electrons, ions)
    # Store new B_pot
    B_pot = np.sum(grid.B**2) / (mu_0 * 2)

    print(f"Setup phase took {time.time() - t_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    t_start = time.time()
    t_last = t_start

    t = 0  # Time in the simulation
    step = 0  # Number of iterations
    while t < params.t_max and step < params.max_iter:
        step += 1

        max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
        dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
        t += dt

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

        # Calculate velocities v^(n+3/2) using the boris pusher
        newton.boris_pusher_1D3V(grid, electrons, dt)
        newton.boris_pusher_1D3V(grid, ions, dt)

        # Calculate densities n_e(x), n_i(x) and rho(x) at full timestep t = n+1
        grid.set_densities(electrons, ions)
        # Calculate E^(n+1)
        maxwell.calc_E_1D3V(grid, dt, params.bc)

        # Calculate current density J^(n+3/2)
        maxwell.calc_curr_dens(grid, electrons, ions)
        # Calculate B^(n+3/2)
        maxwell.calc_B_1D3V(grid, dt, params.bc)

        # Save results every 50 iterations
        if step % 50 == 0:
            KE = electrons.kinetic_energy() + ions.kinetic_energy()
            PE = (grid.dx / 2) * (eps_0 * np.sum(grid.E**2) + (B_pot + np.sum(grid.B**2) / mu_0))
            TE = KE + PE
            results.save(t, KE, PE, TE, grid, electrons, ions)
        # Store new value of B_pot
        B_pot = np.sum(grid.B**2) / mu_0

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

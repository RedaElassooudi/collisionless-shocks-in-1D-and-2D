import time
import numpy as np

import boundary_conditions
from grids import Grid1D3V
import maxwell
import newton
from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import *
from results import Results
from time_constraint import calculate_dt_max


def simulate(electrons: Particles, ions: Particles, params: Parameters):
    np.random.seed(params.seed)
    t_start = time.time()
    if electrons.N != ions.N:
        print("Plasma with non-neutral charge")

    if params.bc is BoundaryCondition.Absorbing:
        assert params.damping_width > 0, "If using absorbing boundary condition, the damping width must be set"

    grid = Grid1D3V(params.x_max, params.dx)
    results = Results()

    max_v = max(np.max(np.linalg.norm(electrons.v, axis=1)), np.max(np.linalg.norm(ions.v, axis=1)))
    dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
    newton.initialize_velocities_half_step_1D3V(grid, electrons, ions, params, dt)

    # Save data at time = 0
    results.save_time(0)
    KE = electrons.kinetic_energy() + ions.kinetic_energy()
    PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
    results.save_energies(KE, PE, KE + PE)
    results.save_densities(grid)
    results.save_phase_space(electrons, ions)
    results.save_fields_ND(grid)
    results.save_cells(grid)

    print(f"Setup phase took {time.time() - t_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    t_start = time.time()
    t_last = t_start

    t = 0  # Time in the simulation
    step = 0  # Number of iterations

    #determine the current density for the first timestep
    maxwell.calc_curr_dens(grid, electrons, ions)
    while t < params.t_max:
        step += 1

        max_v = max(np.max(np.linalg.norm(electrons.v, axis=1)), np.max(np.linalg.norm(ions.v, axis=1)))
        dt = calculate_dt_max(params.dx, max_v, electrons.qm, safety_factor=20)
        t += dt

        # Solve the Maxwell equation on the grid and set the values for J, E, B
        # Store J of the previous timestep
        grid.J_prev = grid.J.copy()
        # Solve for the current density J
        maxwell.calc_curr_dens(grid, electrons, ions)
        """
        # Calculate velocities v^(n+1/2) using Newton's equation
        newton.boris_pusher(grid, electrons, ions, params, dt)
        """
        # Solve the Poisson equation on the grid and set the values for rho, phi and Ex
        maxwell.poisson_solver_1D3V(grid, electrons, ions, params, params.bc)
        # Solve the Ampere and Faraday laws and set values for Ey, Ez, By, Bz
        maxwell.calc_fields(grid, dt, params.bc)

        # Calculate velocities v^(n+1/2) using Newton's equation
        newton.apply_lorenz_force_1D3V(grid, electrons, dt)
        newton.apply_lorenz_force_1D3V(grid, ions, dt, params.bc)
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
            # TODO: 1D -> 1D3V
            boundary_conditions.absorbing_bc_1D(electrons, ions, params.x_max, params.damping_width)
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)

        if step % 50 == 0:
            # Save time in simulation
            results.save_time(t)
            # Energy
            KE_e = electrons.kinetic_energy()
            KE_i = ions.kinetic_energy()
            KE = KE_e + KE_i
            # TODO: not sure about this
            PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
            TE = KE + PE
            results.save_energies(KE, PE, TE)
            # Densities n_e(x), n_i(x)
            results.save_densities(grid)
            # Phase-space (x,v)
            results.save_phase_space(electrons, ions)
            # Electromagnetic fields
            results.save_fields_ND(grid)
            # Shape of the domain (now fixed, will change when using AMR (two cells might join / a cell might split), if we decide to do so)
            results.save_cells(grid)

        if time.time() - t_last > 5:
            t_last = time.time()
            print(f"{step:9}{t:12.4e}{dt:12.4e}{t_last - t_start:21.3e}{TE:14.4e}")

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time() - t_start:21.3e}{TE:14.4e}")
    # TODO: maybe, once we run *A LOT* of iterations, periodically save the data to a file
    # instead of keeping everything in memory
    return results
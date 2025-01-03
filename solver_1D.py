from datetime import datetime
import time
import numpy as np

import boundary_conditions
from grids import Grid1D
import maxwell
import newton
from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import *
from results import Results1D
from time_constraint import calculate_dt_max
from scipy import sparse


def simulate(electrons: Particles, ions: Particles, params: Parameters):
    np.random.seed(params.seed)
    t_start = time.time()
    if electrons.N != ions.N:
        print("Plasma with non-neutral charge")

    if params.bc is BoundaryCondition.Absorbing:
        assert params.damping_width > 0, "If using absorbing boundary condition, the damping width must be set"

    grid = Grid1D(params.x_max, params.dx)
    results = Results1D()

    max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
    dt = calculate_dt_max(params.dx, max_v, electrons.qm, electrons.dimX, safety_factor=20)
    newton.initialize_velocities_half_step_1D(grid, electrons, ions, params, dt)

    # Save data at time = 0
    KE = electrons.kinetic_energy() + ions.kinetic_energy()
    PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
    TE = KE + PE
    results.save(0, KE, PE, TE, grid, electrons, ions)

    print(f"Setup phase took {time.time() - t_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    t_start = time.time()
    t_last = t_start

    # Creates sparse matrix for Thomas solver
    main_diag = -2 * np.ones(grid.n_cells-1)
    off_diag = 1 * np.ones(grid.n_cells-2)
    tridiag = (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    if params.bc is not BoundaryCondition.Periodic:
        tridiag[grid.n_cells-2, grid.n_cells-5:] = np.array([-1, 4, -5, 2])
    tridiag = sparse.csr_matrix(tridiag)

    t = 0  # Time in the simulation
    step = 0  # Number of iterations
    # TODO: We need a proper definition of unit_time, iterating to t = 1 takes EXTREMELY long (using SOR)
    # => What does it mean to simulate until t = 1?
    while t < params.t_max:
        step += 1

        max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
        dt = calculate_dt_max(params.dx, max_v, electrons.qm, electrons.dimX, safety_factor=20)
        t += dt

        # Solve the Poisson equation on the grid and set the values for rho, phi and E
        maxwell.thomas_solver(grid, params.dx, tridiag)

        # Calculate velocities v^(n+1/2) using Newton's equation
        newton.lorenz_force_1D(grid, electrons, dt)
        newton.lorenz_force_1D(grid, ions, dt)

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
            boundary_conditions.absorbing_bc_1D(electrons, ions, params.x_max, params.damping_width)
            newton.advance_positions(electrons, dt)
            newton.advance_positions(ions, dt)

        # Save results every 200 iterations
        if step % 200 == 0:
            KE = electrons.kinetic_energy() + ions.kinetic_energy()
            PE = 0.5 * eps_0 * np.sum(grid.E**2) * params.dx
            TE = KE + PE
            results.save(t, KE, PE, TE, grid, electrons, ions)

        # Log progress every 5 seconds
        if time.time() - t_last > 5:
            t_last = time.time()
            print(f"{step:9}{t:12.4e}{dt:12.4e}{t_last - t_start:21.3e}{TE:14.4e}")

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time() - t_start:21.3e}{TE:14.4e}")
    print("DONE!")
    string_time = datetime.fromtimestamp(t_start).strftime("%Y-%m-%dT%Hh%Mm%Ss")
    results.write(f"Results/{string_time}", params)
    print(f"Results saved in Results/{string_time}/")
    return results

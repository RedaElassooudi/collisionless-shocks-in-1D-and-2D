#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:46:16 2024

@author: u0160596
"""

import time
import numpy as np
import random

import boundary_conditions
import initial_conditions
import newton
import visualizations as vis
import maxwell

# Simulation parameters
num_particles = 20000  # Total number of particles (ions + electrons)
num_cells = 200  # Number of spatial grid cells
num_steps = 1000  # Number of time steps
qm_e = -1.0  # Charge-to-mass ratio for electrons
qm_i = 1.0 / 1836.0  # Charge-to-mass ratio for ions
q_e = -1
q_i = 1
v_te = 1.0  # Thermal velocity for electrons
v_ti = 0.1  # Thermal velocity for ions
dx = 1.0  # Spatial step size
x_max = num_cells * dx  # Maximum position value
damping_width = x_max // 10  # Size of region where dampening will occur
random.seed(42)  # set the random seed
np.random.seed(42)  # Replace 42 with any integer

# Shock generation parameters
bulk_velocity_e = 2.0  # Bulk velocity for electrons (towards left)
bulk_velocity_i = 0.2  # Bulk velocity for ions (towards left)

# Initialize lists for diagnostics
E_x_history = []
density_e_history = []
density_i_history = []
x_e_history = []
v_x_e_history = []
x_i_history = []
v_x_i_history = []
kinetic_energy_history = []
potential_energy_history = []
total_energy_history = []


# Function that determines the optimal timestep based on the CFL condition and the plasma frequency condition
def calculate_max_timestep(dx, v, qm_e):
    # TODO: i'm not fully convinced what the conditions on dt should be using these normalized units
    # CFL condition (particle shouldn't cross more than one cell per timestep)
    dt_cfl = dx / (5.0 * v)  # Factor 5 for safety

    # Plasma frequency condition
    wp = np.sqrt(abs(qm_e))  # Plasma frequency (normalized units)
    dt_wp = 0.2 / wp  # Factor 0.2 for stability

    # Return the more restrictive timestep
    return min(dt_cfl, dt_wp)


# Function to properly initialize velocities for the leapfrog scheme at t-dt/2.
def initialize_velocities_half_step(x_e, v_x_e, x_i, v_x_i, dt, dx, num_cells):
    # TODO: use new rho, phi and E calculations (i.e. place these in functions and call them here)
    # Calculate initial electric field
    rho = np.zeros(num_cells)

    # Charge assignment
    idx_e = (x_e / dx).astype(int) % num_cells
    idx_i = (x_i / dx).astype(int) % num_cells
    np.add.at(rho, idx_e, q_e)
    np.add.at(rho, idx_i, q_i)

    # Solve for initial electric field
    phi = np.zeros(num_cells)
    rho_mean = np.mean(rho)
    rho_tilde = rho - rho_mean

    phi[1:] = np.cumsum(rho_tilde[:-1]) * dx**2

    # Electric field calculation
    E_x = np.zeros(num_cells)
    E_x[:-1] = -(phi[1:] - phi[:-1]) / dx
    E_x[-1] = -(phi[0] - phi[-1]) / dx

    # Get field at particle positions
    E_x_e = E_x[idx_e]
    E_x_i = E_x[idx_i]

    # Push velocities back half timestep
    v_x_e -= 0.5 * qm_e * E_x_e * dt
    v_x_i -= 0.5 * qm_i * E_x_i * dt


# Main simulation function
def run_simulation(bound_cond=0):
    # Initialize particles
    x_e, v_e, x_i, v_i = initial_conditions.initialize_particles(
        num_particles, x_max, bulk_velocity_e, bulk_velocity_i, v_te, v_ti, 3
    )

    # determine the initial timestep
    # Calculate v as the maximum of the norm of the velocities of all particles (e- & i+)
    dt = calculate_max_timestep(
        dx,
        max(np.max(np.linalg.norm(v_e, axis=1)), np.max(np.linalg.norm(v_i, axis=1))),
        qm_e,
    )

    # Initialize velocities at half timestep (v^(-1/2))
    initialize_velocities_half_step(x_e, v_e[:, 0], x_i, v_i[:, 0], dt, dx, num_cells)

    # Spatial grid
    x_grid = np.linspace(0, x_max, num_cells + 1)[:-1]
    rho = np.zeros(num_cells)
    J = np.zeros(num_cells, 3)
    E_x = np.zeros(num_cells)
    E_y = np.zeros(num_cells)
    E_z = np.zeros(num_cells)
    B_x = np.zeros(num_cells)
    B_y = np.zeros(num_cells)
    B_z = np.zeros(num_cells)
    phi = np.zeros(num_cells)

    t_start = time.time()

    for step in range(num_steps):
        # determine the timestep
        dt = calculate_max_timestep(
            dx,
            max(
                np.max(np.linalg.norm(v_e, axis=1)), np.max(np.linalg.norm(v_i, axis=1))
            ),
            qm_e,
        )

        # Position update (x^(n+1) = x^n + v^(n+1/2) * dt)
        # depending on the boundary condition used this needs to be calculated before or after the boudnary conditions have been checked
        if bound_cond == 0 or bound_cond == 2:
            x_e += v_x_e * dt
            x_i += v_x_i * dt

        # Apply boundary conditions
        # check the type of boundary condition to be used
        if bound_cond == 0:
            # remove electrons for x < 0 and x > x_max
            mask_e = (x_e > 0) & (x_e <= x_max)
            v_x_e = v_x_e[mask_e]
            x_e = x_e[mask_e]

            # remove ions beyond for x < 0 and x > x_max
            mask_i = (x_i > 0) & (x_i <= x_max)
            v_x_i = v_x_i[mask_i]
            x_i = x_i[mask_i]

            # check how many electrons were removed
            num_e = len(x_e)
            removed_e = num_particles // 2 - num_e
            if removed_e != 0:
                # if electrons were removed create new electrons by sampling from existing list
                rand_ints_e = random.choices(range(num_e), k=removed_e)
                new_x_e = np.array(
                    [
                        min(
                            max(x_e[i] + np.random.uniform(-dx, dx) * dx, 0),
                            x_max - 0.001,
                        )
                        for i in rand_ints_e
                    ]
                )
                new_v_x_e = np.array([v_x_e[i] for i in rand_ints_e])

                # add these electrons to the old ones
                x_e = np.concatenate((x_e, new_x_e))
                v_x_e = np.concatenate((v_x_e, new_v_x_e))

            # check how many ions were removed
            num_i = len(x_i)
            removed_i = num_particles // 2 - num_i
            if removed_i != 0:
                # if ions were removed create new ions by sampling from existing list
                rand_ints_i = random.choices(range(num_i), k=removed_i)
                new_x_i = np.array(
                    [
                        min(
                            max(x_i[i] + np.random.uniform(-dx, dx) * dx, 0),
                            x_max - 0.001,
                        )
                        for i in rand_ints_i
                    ]
                )
                new_v_x_i = np.array([v_x_i[i] for i in rand_ints_i])

                # add these ions to the old ones
                x_i = np.concatenate((x_i, new_x_i))
                v_x_i = np.concatenate((v_x_i, new_v_x_i))

        elif bound_cond == 1:
            # Apply absorbing boundary conditions at x_max (right boundary)
            boundary_conditions.apply_damping(x_e, v_x_e, x_max, -damping_width)
            boundary_conditions.apply_damping(x_i, v_x_i, x_max, -damping_width)

            # apply absorbing boundary condition at x = 0 (left boundary)
            boundary_conditions.apply_damping(x_e, v_x_e, 0, damping_width)
            boundary_conditions.apply_damping(x_i, v_x_i, 0, damping_width)

            # as we change the velocity profile w.r.t x, computing the new positions must occur after dampening has been calculated
            x_e += v_x_e * dt
            x_i += v_x_i * dt

        elif bound_cond == 2:
            # Apply periodic boundary conditions
            x_e = x_e % x_max
            x_i = x_i % x_max

        # Charge assignment (Cloud in Cell scheme)
        rho.fill(0)
        idx_e = (x_e / dx).astype(int)
        s_e = (x_e / dx) - idx_e
        idx_i = (x_i / dx).astype(int)
        s_i = (x_i / dx) - idx_i
        np.add.at(rho, idx_e, q_e * (1 - s_e))
        np.add.at(rho, (idx_e + 1) % num_cells, q_e * s_e)
        np.add.at(rho, idx_i, q_i * (1 - s_i))
        np.add.at(rho, (idx_i + 1) % num_cells, q_i * s_i)

        # Current density via CIC for all three velocity components
        maxwell.calc_curr_dens(J, v_e, v_i, idx_e, idx_i, s_e, s_i)

        # Solve Poisson's equation
        rho_mean = np.mean(rho)
        rho_tilde = rho - rho_mean

        # Solve for the electric potential using SOR
        # Use phi^n-1 (phi(x) in previous iteration) as initial condition
        # TODO: fix this, iterative solver requires reasonable first guess
        if step == 0:
            phi[1:] = np.cumsum(np.cumsum(rho_tilde[:-1])) * dx**2
        maxwell.solve_poisson_sor(phi, rho_tilde, dx, bound_cond)

        # Electric field calculation
        E_x[:-1] = -(phi[1:] - phi[:-1]) / dx
        E_x[-1] = -(phi[0] - phi[-1]) / dx

        # Gather electric field at particle positions using Cloud-in-Cell weighting
        E_x_e = E_x[idx_e] * (1 - s_e) + E_x[(idx_e + 1) % num_cells] * s_e
        E_x_i = E_x[idx_i] * (1 - s_i) + E_x[(idx_i + 1) % num_cells] * s_i

        # Velocity update (v^(n+3/2) = v^(n+1/2) + q/m * E^(n+1) * dt)
        v_e[:, 0] += qm_e * E_x_e * dt
        v_i[:, 0] += qm_i * E_x_i * dt

        # Compute diagnostics
        density_e = np.zeros(num_cells)
        density_i = np.zeros(num_cells)
        np.add.at(density_e, idx_e, 1)
        np.add.at(density_i, idx_i, 1)

        # Compute energies (use v^(n+1/2) for kinetic energy)
        kinetic_energy_e = 0.5 * np.sum(v_e**2)
        kinetic_energy_i = 0.5 * 1836 * np.sum(v_i**2)
        total_kinetic_energy = kinetic_energy_e + kinetic_energy_i
        potential_energy = 0.5 * np.sum(E_x**2) * dx
        total_energy = total_kinetic_energy + potential_energy

        # Store diagnostics every 10 steps
        if step % 10 == 0:
            E_x_history.append(E_x.copy())
            density_e_history.append(density_e.copy())
            density_i_history.append(density_i.copy())
            x_e_history.append(x_e.copy())
            v_x_e_history.append(v_x_e.copy())
            x_i_history.append(x_i.copy())
            v_x_i_history.append(v_x_i.copy())
            kinetic_energy_history.append(total_kinetic_energy)
            potential_energy_history.append(potential_energy)
            total_energy_history.append(total_energy)

        # Optional: Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")
            print(f"Total Energy: {total_energy:.6f}")
            t_now = time.time()
            print(f"Last 100 steps took: {t_now - t_start} seconds")
            t_start = t_now

    return x_e, v_e, x_i, v_i, E_x_history


# Run the simulation
# bound_cond: 0 = 'Open', 1 = 'Absorbing', 2 = 'Periodic'
x_e, v_x_e, x_i, v_x_i, E_x_history = run_simulation(bound_cond=2)


# Define time steps to plot (start, middle, end)
time_steps = [0, len(E_x_history) // 2, len(E_x_history) - 1]

# Plot results
vis.plot_electric_field(time_steps, E_x_history)
vis.density_profiles_1D(time_steps, density_e_history, density_i_history)
vis.phase_space_1D(time_steps, x_e_history, v_x_e_history, x_i_history, v_x_i_history)
vis.energy_evolution(
    kinetic_energy_history, potential_energy_history, total_energy_history
)

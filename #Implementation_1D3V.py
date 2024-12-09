#Implementation_1D3V

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


def boris_algorithm(particles, E, B, dt):
    """
    Update the particle velocities using the Boris algorithm.
    
    Parameters:
        particles (Particles): The particle object containing positions and velocities.
        E (np.array): Electric field array.
        B (np.array): Magnetic field array.
        dt (float): Time step.
    """
    q = particles.q
    m = particles.m
    v_minus = particles.v + q * E / m * (0.5 * dt)
    t = (q * B / m) * (0.5 * dt)
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    particles.v = v_plus + q * E / m * (0.5 * dt)


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
    newton.initialize_velocities_half_step(grid, electrons, ions, params, dt)

    # Initialize fields
    E = np.zeros(grid.num_points)
    B = np.zeros((grid.num_points, 3))  # Bx, By, Bz (Bx is assumed constant)

    # Time loop
    for step in range(params.num_steps):
        # Update electric and magnetic fields using Maxwell's equations
        E = maxwell.update_electric_field(E, B, electrons, ions, grid, dt)
        B = maxwell.update_magnetic_field(B, E, grid, dt)

        # Update velocities using Boris algorithm
        boris_algorithm(electrons, E, B, dt)
        boris_algorithm(ions, E, B, dt)

        # Update positions
        newton.update_positions(grid, electrons, dt)
        newton.update_positions(grid, ions, dt)

        # Apply boundary conditions
        boundary_conditions.apply(grid, electrons, params.bc)
        boundary_conditions.apply(grid, ions, params.bc)

        # Diagnostics and results storage
        results.store(electrons, ions, E, B, step)

    t_end = time.time()
    print(f"Simulation completed in {t_end - t_start} seconds.")
    results.finalize()


def update_electric_field(E, B, electrons, ions, grid, dt):
    """
    Update the electric field using Faraday's Law and charge density.
    
    Parameters:
        E (np.array): Electric field array.
        B (np.array): Magnetic field array.
        electrons (Particles): Electrons particle data.
        ions (Particles): Ions particle data.
        grid (Grid1D3V): Grid data for the simulation.
        dt (float): Time step.

    Returns:
        np.array: Updated electric field array.
    """
    # Calculate the current density from particle velocities and positions
    J = grid.calculate_current_density(electrons, ions)
    # Use Ampere's law to update E
    E += dt * (J - np.gradient(B[:, 1], grid.dx))  # Update Ex based on current density and magnetic field gradients
    return E


def update_magnetic_field(B, E, grid, dt):
    """
    Update the magnetic field using Faraday's Law.
    
    Parameters:
        B (np.array): Magnetic field array.
        E (np.array): Electric field array.
        grid (Grid1D3V): Grid data for the simulation.
        dt (float): Time step.

    Returns:
        np.array: Updated magnetic field array.
    """
    # Use Faraday's law to update B
    B[:, 1] -= dt * np.gradient(E, grid.dx)  # Update By based on Ex
    B[:, 2] -= dt * np.gradient(E, grid.dx)  # Update Bz based on Ex
    return B

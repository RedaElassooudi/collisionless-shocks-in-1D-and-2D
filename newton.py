import numpy as np

from grids import Grid1D, Grid1D3V
import maxwell
from parameters import Parameters
from particles import Particles


def advance_positions(particles: Particles, dt):
    """
    Perform position update: x^(n+1) = x^n + v^(n+1/2) * dt
    Works for 1D, 1D3V, 2D, 3D
    """
    particles.x += particles.v[:, 0 : particles.dimX] * dt


def initialize_velocities_half_step_1D(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, dt: float):
    """
    Function to properly initialize velocities for the leapfrog scheme at t-dt/2.
    """
    # Calculate initial electric field
    maxwell.poisson_solver_1D(grid, electrons, ions, params, first=True)
    # Apply Lorenz force backwards in time to find v^(-1/2)
    apply_lorenz_force_1D(grid, electrons, -dt / 2)
    apply_lorenz_force_1D(grid, ions, -dt / 2)


def apply_lorenz_force_1D(grid: Grid1D, particles: Particles, dt):
    # Get field at particle positions
    E = (
        grid.E[particles.idx] * (1 - particles.cic_weights)
        + grid.E[(particles.idx + 1) % grid.n_cells] * particles.cic_weights
    )
    # Push velocities back half timestep
    particles.v += particles.qm * E * dt


def boris_pusher_1D3V(grid: Grid1D3V, particles: Particles, dt):
    # Get field at particle positions
    E = (
        grid.E[particles.idx.flatten()] * (1 - particles.cic_weights)
        + grid.E[(particles.idx.flatten() + 1) % grid.n_cells] * particles.cic_weights
    )
    B = (
        grid.B[particles.idx.flatten()] * (1 - particles.cic_weights)
        + grid.B[(particles.idx.flatten() + 1) % grid.n_cells] * particles.cic_weights
    )
    # Half step due to electric field
    particles.v += particles.qm * E * dt / 2

    # Full step due to magnetic field
    t = particles.qm * B * dt / 2
    t_mag2 = np.einsum('ij,ij->i', t, t)
    t_mag2 = t_mag2[:, np.newaxis]
    s = (2 * t) / (1+ t_mag2)
    particles.v += np.cross(particles.v + np.cross(particles.v, t), s)

    # Half step due to electric field
    particles.v +=  particles.qm * E * dt / 2



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
    maxwell.poisson_solver(grid, electrons, ions, params, first=True)
    # Apply Lorenz force backwards in time to find v^(-1/2)
    lorenz_force_1D(grid, electrons, -dt / 2)
    lorenz_force_1D(grid, ions, -dt / 2)


def lorenz_force_1D(grid: Grid1D, particles: Particles, dt):
    # Get field at particle positions
    E = grid.E[particles.idx] * (1 - particles.cic_weights) + grid.E[(particles.idx + 1) % grid.n_cells] * particles.cic_weights
    # Update velocities using 1D Lorenz force formulation
    particles.v += particles.qm * E * dt


def initialize_velocities_half_step_1D3V(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, dt: float):
    # Apply Lorenz force backwards in time to find v^(-1/2)
    boris_pusher_1D3V(grid, electrons, dt / 2)
    boris_pusher_1D3V(grid, ions, dt / 2)


def boris_pusher_1D3V(grid: Grid1D3V, particles: Particles, dt):
    # extra source: https://www.particleincell.com/2011/vxb-rotation/
    # Get field at particle positions
    E = (
        grid.E[particles.idx.flatten()] * (1 - particles.cic_weights)
        + grid.E[(particles.idx.flatten() + 1) % grid.n_cells] * particles.cic_weights
    )
    B = (
        grid.B[particles.idx.flatten()] * (1 - particles.cic_weights)
        + grid.B[(particles.idx.flatten() + 1) % grid.n_cells] * particles.cic_weights
    )
    # Calculate v⁻ = v_n + 𝜖
    particles.v += particles.qm * E * dt / 2

    beta = particles.qm * B * dt / 2
    # einsum really is the fastest: https://stackoverflow.com/a/45006970/15836556
    beta_sq = np.einsum("ij,ij->i", beta, beta)
    beta_sq = beta_sq[:, np.newaxis]
    s = (2 * beta) / (1 + beta_sq)
    # Calculate v⁻ + (v⁻ + (v⁻ × β)) × s
    particles.v += np.cross(particles.v + np.cross(particles.v, beta), s)

    # v_n+1 = previous + 𝜖
    particles.v += particles.qm * E * dt / 2

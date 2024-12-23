import numpy as np

from grids import Grid1D, Grid1D3V, Grid2D
import maxwell
from parameters import Parameters
from particles import Particles


def advance_positions(particles: Particles, dt):
    """
    Perform position update: x^(n+1) = x^n + v^(n+1/2) * dt
    Works for 1D, 1D3V, 2D, 3D
    """
    particles.x += particles.v[:, 0 : particles.dimX] * dt


def initialize_velocities_half_step_1D(
    grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, dt: float
):
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
    E = (
        grid.E[particles.idx] * (1 - particles.cic_weights)
        + grid.E[(particles.idx + 1) % grid.n_cells] * particles.cic_weights
    )
    # Update velocities using 1D Lorenz force formulation
    particles.v += particles.qm * E * dt


def initialize_velocities_half_step_1D3V(grid: Grid1D3V, electrons: Particles, ions: Particles, params: Parameters, dt: float):
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
        grid.B[particles.idx_staggered.flatten()] * (1 - particles.cic_weights_staggered)
        + grid.B[(particles.idx_staggered.flatten() + 1) % grid.n_cells] * particles.cic_weights_staggered
    )

    # Calculate v⁻ = v_n + 𝜖
    particles.v += particles.qm * E * dt / 2

    beta = particles.qm * B * dt / 2
    beta_sq = np.einsum("ij,ij->i", beta, beta)
    beta_sq = beta_sq[:, np.newaxis]
    s = (2 * beta) / (1 + beta_sq)
    # Calculate v⁻ + (v⁻ + (v⁻ × β)) × s
    # v_p = v⁻ + (v⁻ × β)
    v_p = particles.v + np.cross(particles.v, beta)
    v_p = particles.v + np.concatenate((particles.v[:,1:2] * beta[:,2:] - particles.v[:,2:] * beta[:,1:2],
                                        particles.v[:,2:] * beta[:,:1] - particles.v[:,:1] * beta[:,2:] ,
                                        particles.v[:,:1] * beta[:,1:2] - particles.v[:,1:2] * beta[:,:1]), axis=1)
    #v⁻ + (v_p) × s
    particles.v += np.cross(v_p, s)
    particles.v += np.concatenate((v_p[:,1:2] * s[:,2:] - v_p[:,2:] * s[:,1:2],
                                        v_p[:,2:] * s[:,:1] - v_p[:,:1] * s[:,2:] ,
                                        v_p[:,:1] * s[:,1:2] - particles.v[:,1:2] * s[:,:1]), axis=1)

    # v_n+1 = previous + 𝜖
    particles.v += particles.qm * E * dt / 2


def initialize_velocities_half_step_2D(grid: Grid2D, electrons: Particles, ions: Particles, params: Parameters, dt: float):
    # Apply Lorenz force backwards in time to find v^(-1/2)
    boris_pusher_2D(grid, electrons, dt / 2)
    boris_pusher_2D(grid, ions, dt / 2)


def boris_pusher_2D(grid: Grid2D, particles: Particles, dt):
    # extra source: https://www.particleincell.com/2011/vxb-rotation/
    # Create arrays to get adjacent cell coordinates
    x_adj = np.zeros((particles.N, 2), dtype=int)
    y_adj = np.zeros((particles.N, 2), dtype=int)
    x_adj[:,0] = 1
    y_adj[:,1] = 1
    # Determine arrays for adjacent cell coordinates
    upper_x = (particles.idx + x_adj ) % grid.n_cells
    upper_y = (particles.idx + y_adj) % grid.n_cells
    upper_xy = (particles.idx + x_adj + y_adj) % grid.n_cells
    # Get field at particle positions
    E = (
        grid.E[particles.idx[:,0], particles.idx[:,1]] * ((1 - particles.cic_weights[:,0]) * (1 - particles.cic_weights[:,1]))[:, np.newaxis]
        + grid.E[upper_x[:,0], upper_x[:,1]] * (particles.cic_weights[:,0] * (1 - particles.cic_weights[:,1]))[:, np.newaxis]
        + grid.E[upper_y[:,0], upper_y[:,1]] * (particles.cic_weights[:,1] * (1 - particles.cic_weights[:,0]))[:, np.newaxis]
        + grid.E[upper_xy[:,0], upper_xy[:,1]] * (particles.cic_weights[:,1] * particles.cic_weights[:,0])[:, np.newaxis]
    )
    B = (
        grid.B[particles.idx[:,0], particles.idx[:,1]] * ((1 - particles.cic_weights[:,0]) * (1 - particles.cic_weights[:,1]))[:, np.newaxis]
        + grid.B[upper_x[:,0], upper_x[:,1]] * (particles.cic_weights[:,0] * (1 - particles.cic_weights[:,1]))[:, np.newaxis]
        + grid.B[upper_y[:,0], upper_y[:,1]] * (particles.cic_weights[:,1] * (1 - particles.cic_weights[:,0]))[:, np.newaxis]
        + grid.B[upper_xy[:,0], upper_xy[:,1]] * (particles.cic_weights[:,1] * particles.cic_weights[:,0])[:, np.newaxis]
    )
    # Calculate v⁻ = v_n + 𝜖
    particles.v += particles.qm * E * dt / 2

    beta = particles.qm * B * dt / 2
    beta_sq = np.sum(beta * beta, axis=1, keepdims=True)
    s = (2 * beta) / (1 + beta_sq)
    # Calculate v⁻ + (v⁻ + (v⁻ × β)) × s
    # v_p = v⁻ + (v⁻ × β)
    v_p = particles.v + np.concatenate((particles.v[:,1:] * beta, -particles.v[:,:1] * beta), axis=1)
    # v = v⁻ + v_p × s
    particles.v += np.concatenate((v_p[:,1:] * s, -v_p[:,:1] * s), axis=1)

    # v_n+1 = previous + 𝜖
    particles.v += particles.qm * E * dt / 2
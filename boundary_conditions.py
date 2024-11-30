# Modify x, v (but possibly also E, B?) based on boundary conditions
import numpy as np

from particles import Particles


def periodic_bc(electrons: Particles, ions: Particles, x_max: float):
    """
    Apply periodic boundary conditions to electrons and ions that have left the domain

    Parameters:
        electrons (Particles): the electrons
        ions (Particles): the ions
        x_max (float / tuple): the size of the domain. float for 1D, tuple for 2D and 3D
    """  
    _periodic_bc(electrons, x_max)
    _periodic_bc(ions, x_max)


def _periodic_bc(particles: Particles, x_max: float):
    particles.x %= x_max


def open_bc(electrons: Particles, ions: Particles, x_max: float, dx: float):
    """
    Apply open boundary conditions, works for 1D, 2D and 3D :3

    Remove all particles outside of the domain [0; x_max].
    Also works for 2D and 3D, so domain = [0, x_max[0]] x [0, x_max[1]] (x [0, x_max[2]])
    Insert new particles drawn randomly
    Maybe an interesting alternative approach?:
    https://pubs.aip.org/aip/pop/article/15/8/082102/897118/Particle-in-cell-simulation-of-collisionless
    @Simon, check this:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JA019282
    """
    _open_bc(electrons, x_max, dx)
    _open_bc(ions, x_max, dx)


def _open_bc(particles: Particles, x_max: float, dx: float):
    # Remember initial number of particles
    N_0 = particles.N
    # Filter particles, only keep the ones inside domain
    mask = (particles.x > 0) & (particles.x <= x_max)
    mask = mask.all(axis=1)  # Make sure the boolean expression is true for all coordinates of the point x
    particles.filter(mask)
    # Count how many particles were removed
    N_removed = N_0 - particles.N
    if N_removed != 0:
        # if particles were removed, replace them by sampling from existing ones
        new_idxs = np.random.randint(particles.N, size=N_removed)
        new_x = particles.x[new_idxs] + np.random.uniform(-dx, dx, (N_removed, particles.dimX))
        new_x = np.minimum(np.maximum(new_x, 0), x_max - dx / 10)
        new_v = particles.v[new_idxs]

        # add these electrons to the old ones
        particles.add_particles(new_x, new_v)

#When using Absorbiing boundaries, make sure that the initial conditions are chosen such that there are no particles near the edges
def absorbing_bc_1D(electrons: Particles, ions: Particles, x_max: float, damping_width: float):
    # Apply absorbing boundary conditions at x_max (right boundary)
    _apply_damping(electrons, x_max, -damping_width)
    _apply_damping(ions, x_max, -damping_width)

    # apply absorbing boundary condition at x = 0 (left boundary)
    _apply_damping(electrons, 0, damping_width)
    _apply_damping(ions, 0, damping_width)


def _apply_damping(par: Particles, x_boundary: float, width: float, factor=0.1):
    # differentiate between the layers at x = 0 and x = x_max
    if width > 0:
        damping_region = par.x <= x_boundary + width
        damp_factor = factor * min(par.v) / (x_boundary - width) ** 2
        par.v[damping_region] += abs(damp_factor * (par.x[damping_region] - width) ** 2)
    else:
        damping_region = par.x >= x_boundary + width
        damp_factor = factor * max(par.v) / (x_boundary - width) ** 2
        par.v[damping_region] -= abs(damp_factor * (par.x[damping_region] - width) ** 2)
    return

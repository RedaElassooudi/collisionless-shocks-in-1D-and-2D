import numpy as np

from particles import Particles
from physical_constants import *


# Initialize particle positions and velocities with a bulk flow
def initialize_particles(num_particles: int, x_max: float, v_bulk_e: float, v_bulk_i: float, dim: int):
    # Electrons (uniformly distributed)
    num_electrons = num_particles // 2
    electrons = Particles(num_electrons, 1, dim, m_e, q_e)
    np.copyto(electrons.x, np.random.uniform(0, x_max, (num_electrons, 1)))
    np.copyto(electrons.v, np.random.normal(v_bulk_e, v_te, (num_electrons, dim)))

    # Ions (density gradient)
    num_ions = num_particles - num_electrons
    ions = Particles(num_ions, 1, dim, m_i, q_i)
    num_ions_left = int(num_ions * 0.7)  # 70% ions on the left half
    num_ions_right = num_ions - num_ions_left  # Remaining ions on the right half

    x_i_left = np.random.uniform(0, x_max / 2, (num_ions_left, 1))
    x_i_right = np.random.uniform(x_max / 2, x_max, (num_ions_right, 1))
    np.copyto(ions.x, np.concatenate((x_i_left, x_i_right)))
    np.copyto(ions.v, np.random.normal(v_bulk_i, v_ti, (num_ions, dim)))

    return electrons, ions


def two_stream(num_particles: int, x_max: float, v_the: float, v_bulk: float, nx: int, eps: float, mode: int):
    """
    Initializes a two-stream instability setup with electron and ion particles.

    Parameters:
    - num_particles (int): Total number of particles (electrons + ions).
    - x_max (float): Length of the spatial domain.
    - v_the (float): Thermal velocity of the electrons.
    - v_bulk (float): Bulk drift velocity of the electrons.
    - nx (int): Number of grid points in the spatial domain.
    - eps (float): Amplitude of the initial sinusoidal perturbation.
    - mode (int): Mode number of the perturbation (number of wavelengths in the domain).

    Returns:
    - tuple: A tuple containing `electrons` and `ions` as `Particles` objects.
    """
    # Charge neutrality: n_e = n_i
    num_electrons = num_ions = num_particles // 2

    electrons = Particles(num_electrons, 1, 3, m_e, q_e)
    # endpoint = False makes sure the last particle has position x_n < x_max
    electrons.x = np.linspace(0, (1 - (1 / (2 * nx))) * x_max, num_electrons, endpoint=False).reshape(num_electrons, 1)

    electrons.x += eps * np.cos(2 * np.pi * mode / x_max * electrons.x)
    # Put particles which left the domain due to perturbation back inside the domain
    electrons.x[electrons.x < 0] += x_max
    electrons.x[electrons.x >= x_max] -= x_max

    # Only two streams in x directions
    electrons.v = v_the * np.random.randn(num_electrons, 3)
    pm = np.arange(num_electrons)
    pm = 1 - 2 * np.mod(pm + 1, 2)
    electrons.v[:, 0] += pm * v_bulk  # Drift plus thermal spread

    ions = Particles(num_ions, 1, 3, m_i, q_i)
    ions.x = np.linspace(0, (1 - (1 / (2 * nx))) * x_max, num_ions, endpoint=False).reshape(num_ions, 1)
    ions.v.fill(0)

    return electrons, ions

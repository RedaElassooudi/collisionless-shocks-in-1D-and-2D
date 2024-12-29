import numpy as np

from particles import Particles
from physical_constants import *


# Initialize particle positions and velocities with a bulk flow
def initialize_particles(num_particles: int, x_max: float, v_bulk_e: float, v_bulk_i: float, dimx: int, dimv: int):
    # Electrons (uniformly distributed)
    num_electrons = num_particles // 2
    electrons = Particles(num_electrons, dimx, dimv, m_e, q_e)
    np.copyto(electrons.x, np.random.uniform(0, x_max, (num_electrons, dimx)))
    np.copyto(electrons.v, np.random.normal(v_bulk_e, v_te, (num_electrons, dimv)))

    # Ions (density gradient)
    num_ions = num_particles - num_electrons
    ions = Particles(num_ions, dimx, dimv, m_i, q_i)
    num_ions_left = int(num_ions * 0.7)  # 70% ions on the left half
    num_ions_right = num_ions - num_ions_left  # Remaining ions on the right half

    x_i_left = np.random.uniform(0, x_max / 2, (num_ions_left, dimx))
    x_i_right = np.random.uniform(x_max / 2, x_max, (num_ions_right, dimx))
    np.copyto(ions.x, np.concatenate((x_i_left, x_i_right)))
    np.copyto(ions.v, np.random.normal(v_bulk_i, v_ti, (num_ions, dimv)))

    electrons.v_to_u()
    ions.v_to_u()
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
    electrons.x = np.linspace(0, x_max, num_electrons, endpoint=False).reshape(num_electrons, 1)

    electrons.x += eps * np.cos(2 * np.pi * mode / x_max * electrons.x)
    # Put particles which left the domain due to perturbation back inside the domain
    electrons.x[electrons.x < 0] += x_max
    electrons.x[electrons.x >= x_max] -= x_max

    # Only two streams in x directions
    electrons.v = v_the * np.random.randn(num_electrons, 3)
    pm = np.arange(num_electrons)
    pm = 1 - 2 * np.mod(pm + 1, 2)
    electrons.v[:, 0] += pm * v_bulk  # Drift plus thermal spread
    electrons.v_to_u()

    ions = Particles(num_ions, 1, 3, m_i, q_i)
    ions.x = np.linspace(0, x_max, num_ions, endpoint=False).reshape(num_ions, 1)
    ions.v.fill(0)
    ions.v_to_u()
    return electrons, ions


def shock_tube(num_particles: int, x_max: float, v_the: float, v_bulk: float):
    """
    All particles have random individual thermal speed drawn from Boltzmann distribution with v_the = v_thi
    - 50% electrons in [0, x/2) with average speed `v_bulk`,
    - 50% ions in [0, x/2) with average speed `v_bulk`,
    - 50% electrons in [x/2, x_max) with zero average speed,
    - 50% ions in [x/2, x_max) with zero average speed,

    Parameters:
    - num_particles (int): Total number of particles (electrons + ions).
    - x_max (float): Length of the spatial domain.
    - v_the (float): Thermal velocity of the electrons.
    - v_bulk (float): Bulk drift velocity of the electrons.

    Returns:
    - tuple: A tuple containing `electrons` and `ions` as `Particles` objects.
    """
    # For charge neutrality, split half electrons, half ions
    num_electrons = num_ions = num_particles // 2

    num_e_left = num_electrons // 2
    num_e_right = num_electrons - num_e_left
    num_i_left = num_ions // 2
    num_i_right = num_ions - num_i_left

    # Create electron & ion containers
    electrons = Particles(num_electrons, 1, 3, m_e, q_e)
    ions = Particles(num_ions, 1, 3, m_i, q_i)

    electrons_left_x = np.random.uniform(0, x_max / 2, (num_e_left, 1))
    electrons_right_x = np.random.uniform(x_max / 2, x_max, (num_e_right, 1))
    electrons.x = np.vstack((electrons_left_x, electrons_right_x))

    electrons.v = v_the * np.random.randn(num_electrons, 3)
    # Left electrons have E(v_x) = v_bulk, right electrons have E(v_x) = 0
    electrons.v[0:num_e_left, 0] += v_bulk

    ions_left_x = np.random.uniform(0, x_max / 2, (num_i_left, 1))
    ions_right_x = np.random.uniform(x_max / 2, x_max, (num_i_right, 1))
    ions.x = np.vstack((ions_left_x, ions_right_x))

    ions.v = v_the * np.random.randn(num_ions, 3)
    # Left ions have E(v_x) = v_bulk, right ions have E(v_x) = 0: same as electrons
    ions.v[0:num_i_left, 0] += v_bulk

    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions


def bump_on_tail(num_particles: int, x_max: float, v_the: float, v_bump: float, eps: float, mode: int, alpha: float = 0.05):
    # Split total particles into electrons and ions equally
    num_electrons = num_ions = num_particles // 2

    electrons = Particles(num_electrons, 1, 3, m_e, q_e)

    electrons.x = np.random.uniform(0, x_max, (num_electrons, 1))

    # TODO: which initial conditions does bump on tail represent?
    # XP1 = 0.0001
    # mode = 3
    electrons.x += eps * np.cos(2.0 * np.pi * mode / x_max * electrons.x)
    #  periodic BC
    electrons.x[electrons.x < 0] += x_max
    electrons.x[electrons.x >= x_max] -= x_max

    # Number of electrons in bump vs. bulk
    n_bump = int(alpha * num_electrons)
    n_bulk = num_electrons - n_bump

    # Bulk population ~ Maxwellian centered at 0
    v_bulk_array = v_the * np.random.randn(n_bulk, 3)

    # Bump population ~ Maxwellian centered at +v_bump in the x-direction only
    v_bump_array = v_the * np.random.randn(n_bump, 3)
    v_bump_array[:, 0] += v_bump  # shift in +x direction

    # Combine bulk + bump
    combined_v = np.vstack([v_bulk_array, v_bump_array])
    np.random.shuffle(combined_v)  # shuffle so they mix randomly

    # Assign to electron velocities
    electrons.v = combined_v

    ions = Particles(num_ions, 1, 3, m_i, q_i)
    ions.x = np.random.uniform(0, x_max, (num_ions, 1))
    ions.v.fill(0.0)

    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions

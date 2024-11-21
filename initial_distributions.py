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

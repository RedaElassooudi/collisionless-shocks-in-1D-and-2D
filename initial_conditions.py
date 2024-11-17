import numpy as np


# Initialize particle positions and velocities with a bulk flow
def initialize_particles(num_particles, x_max, v_bulk_e, v_bulk_i, v_te, v_ti, dim):
    # Electrons (uniformly distributed with bulk velocity towards left)
    num_electrons = num_particles // 2
    x_e = np.random.uniform(0, x_max, num_electrons)
    v_e = np.random.normal(v_bulk_e, v_te, (num_electrons, dim))

    # Ions (density gradient with bulk velocity towards left)
    num_ions = num_particles - num_electrons
    num_ions_left = int(num_ions * 0.7)  # 70% ions on the left half
    num_ions_right = num_ions - num_ions_left  # Remaining ions on the right half

    x_i_left = np.random.uniform(0, x_max / 2, num_ions_left)
    x_i_right = np.random.uniform(x_max / 2, x_max, num_ions_right)
    x_i = np.concatenate((x_i_left, x_i_right))
    v_i = np.random.normal(v_bulk_i, v_ti, (num_ions, dim))

    return x_e, v_e, x_i, v_i

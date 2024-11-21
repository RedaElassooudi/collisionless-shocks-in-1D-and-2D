import time
import numpy as np
from math import ceil

import maxwell
from parameters import Parameters
from particles import Particles
from physical_constants import *
from time_constraint import calculate_dt_max


def simulate(electrons: Particles, ions: Particles, params: Parameters):
    max_v = max(np.max(np.linalg.norm(electrons.v, axis=1)), np.max(np.linalg.norm(ions.v, axis=1)))
    assert params.dt < (
        dt_max := calculate_dt_max(params.dx, max_v)
    ), f"params.dt should be smaller than (Delta t)_max = {dt_max}"

    num_steps = ceil(params.t_max / params.dt)
    num_cells = ceil(params.x_max / params.dx)

    J = np.zeros((num_cells, 3))
    E = np.zeros((num_cells, 3))
    B = np.zeros((num_cells, 3))
    rho = np.zeros((1, 1))

    t_start = time.time()

    for step in range(num_steps):
        rho.fill(0)
        idx_e = (electrons.x / params.dx).astype(int)
        s_e = (electrons.x / params.dx) - idx_e
        idx_i = (ions.x / params.dx).astype(int)
        s_i = (ions.x / params.dx) - idx_i
        np.add.at(rho, idx_e, q_e * (1 - s_e))
        np.add.at(rho, (idx_e + 1) % num_cells, q_e * s_e)
        np.add.at(rho, idx_i, q_i * (1 - s_i))
        np.add.at(rho, (idx_i + 1) % num_cells, q_i * s_i)

        # Current density via CIC for all three velocity components
        maxwell.calc_curr_dens(J, electrons.v, ions.v, idx_e, idx_i, s_e, s_i)

        # Solve Poisson's equation
        rho_mean = np.mean(rho)
        rho_tilde = rho - rho_mean

        # Solve for the electric potential using SOR
        # Use phi^n-1 (phi(x) in previous iteration) as initial condition
        # TODO: fix this, iterative solver requires reasonable first guess
        if step == 0:
            phi[1:] = np.cumsum(np.cumsum(rho_tilde[:-1])) * dx**2
        maxwell.solve_poisson_sor(phi, rho_tilde, dx, bound_cond)

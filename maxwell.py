# Contains Maxwell solver / Poisson solver
# Solving Poisson's equation:
#   - https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#   - https://en.wikipedia.org/wiki/Relaxation_(iterative_method)
#   - https://en.wikipedia.org/wiki/Successive_over-relaxation
import numpy as np
from numba import jit


def naive_poisson_solver(rho):
    pass


@jit
def thomas_solver(a, b, c, d):
    """
    Can Thomas' algorithm be used to solve our 1D case of Poisson's equation?
    code source: https://stackoverflow.com/questions/8733015/
    Solve Ax = d, where A = tri(a, b, c), i.e. A is tridiagonal with a on the subdiagonal,
    b on the diagonal and c on the upper diagonal

    In our case, a = 1, b = -2, c = 1 and d = -rho * dx^2

    Boundary conditions will require special care!!
    To keep tridiagonal structure, maybe use phi(x_0/N, t_n-1) for the values at the
    boundaries calculated at t_n?
    """
    n = len(d)
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


# TODO: if we keep using this solver, try numba
def solve_poisson_sor(
    phi, rho_tilde, dx, bound_cond, max_iter=1000, tol=1e-6, omega=1.5
):  # max_iter is number of iterations, tol is error tollerance, omega is relaxation factor
    """
    Update for the SOR process for the electric field
    """
    num_cells = len(rho_tilde)

    # choice of boundary conditions determines the range over which this solver is applied
    if bound_cond == 0 or bound_cond == 1:
        for _ in range(max_iter):
            max_error = 0
            for i in range(1, num_cells - 1):  # Inner grid points
                old_phi = phi[i]
                phi[i] = (1 - omega) * phi[i] + (omega / 2) * (
                    phi[i + 1] + phi[i - 1] + dx**2 * rho_tilde[i]
                )
                max_error = max(max_error, abs(phi[i] - old_phi))

            # Apply boundary conditions to calculate the edge values up to third order accuracy using interpolation
            phi[0] = 3 * phi[1] - 3 * phi[2] + phi[3]
            phi[-1] = 3 * phi[-2] - 3 * phi[-3] + phi[-4]
            if max_error < tol:
                print("SOR Converged before maximum iterations!! I AM HAPPY!!")
                break
    else:
        for _ in range(max_iter):
            max_error = 0
            for i in range(0, num_cells):  # Inner grid points
                old_phi = phi[i - 1]
                phi[i - 1] = (1 - omega) * phi[i - 1] + (omega / 2) * (
                    phi[i] + phi[i - 2] + dx**2 * rho_tilde[i - 1]
                )
                max_error = max(max_error, abs(phi[i - 1] - old_phi))
            if max_error < tol:
                print("SOR Converged before maximum iterations!! I AM HAPPY!!")
                break
    return


def calc_curr_dens(J, v_e, v_i, idx_e, idx_i, s_e, s_i, q_e, q_i):
    # Current density via CIC for all three velocity components
    num_cells = np.size(J, axis=0)
    J.fill(0)
    np.add.at(J, idx_e, v_e[idx_e] * q_e * (1 - s_e))
    np.add.at(J, (idx_e + 1) % num_cells, v_e[idx_e] * q_e * s_e)
    np.add.at(J, idx_i, v_i[idx_i] * q_i * (1 - s_i))
    np.add.at(J, (idx_i + 1) % num_cells, v_i[idx_i] * q_i * s_i)

    return

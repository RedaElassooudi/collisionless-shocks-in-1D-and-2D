# Contains Maxwell solver / Poisson solver
# Solving Poisson's equation:
#   - https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#   - https://en.wikipedia.org/wiki/Relaxation_(iterative_method)
#   - https://en.wikipedia.org/wiki/Successive_over-relaxation
import numpy as np
from numba import jit

from grids import Grid1D, Grid1D3V
from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import *


def poisson_solver(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, first=False):
    grid.set_densities(electrons, ions)

    # TODO: commented out the SOR solver so that we can have a result.
    # It is very very slow since it converges way to slowly.
    # See bottom of the file for some timing comparisons
    # Also, another (bigger) issue is that v increases a lot for some reason with the SOR solver,
    # which makes dt very very small and requires even more iterations

    # If this is the first time we solve the poisson equations,
    # we need a good initial guess for the iterative SOR solver,
    # else convergence can't be achieved within a reasonable number of iterations.
    # if first:
    #     naive_poisson_solver(grid, params.dx)
    # solve_poisson_sor(grid.phi, -grid.rho / eps_0, params.dx, params.bc, params.SOR_max_iter, params.SOR_tol, params.SOR_omega)
    naive_poisson_solver(grid, params.dx)

    # Electric field calculation
    grid.E[:-1] = -(grid.phi[1:] - grid.phi[:-1]) / params.dx + grid.E_0[:-1]
    # take boundary conditions into account
    if params.bc is BoundaryCondition.Periodic:
        # Warning: you will get an (unphysically) large field at the right
        # boundary if phi(x) is not periodic
        grid.E[-1] = -(grid.phi[0] - grid.phi[-1]) / params.dx + grid.E_0[-1]
    else:  # use second order interpolation to get the last value
        grid.E[-1] = 2 * grid.E[-3] - grid.E[-2] + grid.E_0[-1]


def naive_poisson_solver(grid: Grid1D, dx: float):
    """
    ∇∙E = ρ(x)/ε & E(x) = -∇ɸ(x) ⇒ ∆ɸ(x) = -ρ(x)/ε ⇒ d^2ɸ(x)/dx^2 = -ρ(x)/ε ⇒
    ɸ(x) = -1/ε * ∬ρ(x) + C_1*x + C_2
    """
    # TODO: we should probably determine phi(0) and phi(L) in some way such that we can define C_1 and C_2
    # However, C_2 can probably be set to zero, as potentials are relative.
    # It might be possible to use higher order formulas to approximate the double integral?
    grid.phi.fill(0)
    grid.phi[1:] = -1 * np.cumsum(np.cumsum(grid.rho[:-1])) * dx**2 / eps_0


@jit
def solve_poisson_sor(u, f, dx, bound_cond, max_iter=100000, tol=1e-4, omega=1.5):
    """
    Solve the Poisson equation: ∆u(x) = f(x)

    The 1D Gauss equation (∇∙E = ρ(x)/ε & E(x) = -∇ɸ(x)) has f(x) = -ρ(x)/ε

    Uses finite differences and solves the resulting linear system using Successive Over-Relaxation (SOR)
    """
    num_cells = np.size(f)

    if bound_cond is BoundaryCondition.Open or bound_cond is BoundaryCondition.Absorbing:
        for _ in range(max_iter):
            max_diff = 0
            for i in range(1, num_cells - 1):  # Inner grid points
                old_u = u[i]
                u[i] = (1 - omega) * u[i] + (omega / 2) * (u[i + 1] + u[i - 1] - dx**2 * f[i])
                max_diff = max(max_diff, abs(u[i] - old_u))

            # Open and absorbing boundaries have no explicit formulation for the fields
            # at the boundaries, so we "make up" boundary conditions by extrapolating u(x)
            u[0] = 3 * u[1] - 3 * u[2] + u[3]
            u[-1] = 3 * u[-2] - 3 * u[-3] + u[-4]
            if max_diff < tol:
                print("SOR Converged before maximum iterations!! I AM HAPPY!!")
                break
    else:  # Periodic boundaries
        for _ in range(max_iter):
            max_diff = 0
            for i in range(num_cells - 1):  # Inner grid points
                old_u = u[i]
                u[i] = (1 - omega) * u[i] + (omega / 2) * (u[i + 1] + u[i - 1] - dx**2 * f[i])
                max_diff = max(max_diff, abs(u[i] - old_u))
            u[-1] = (1 - omega) * u[-1] + (omega / 2) * (u[0] + u[-2] - dx**2 * f[-1])
            if max_diff < tol:
                print("SOR Converged before maximum iterations!! I AM HAPPY!!")
                break


# TODO
# @jit
def thomas_solver(a, b, c, d):
    """
    Can Thomas' algorithm be used to solve our 1D case of Poisson's equation?
    code source: https://stackoverflow.com/questions/8733015/
    Solve Ax = d, where A = tri(a, b, c), i.e. A is tridiagonal with a on the subdiagonal,
    b on the diagonal and c on the upper diagonal

    In our case, a = 1, b = -2, c = 1 and d = -rho/eps * dx^2

    Boundary conditions will require special care!!
    To keep tridiagonal structure, maybe use phi(x, t_n-1) for the values which would
    break the structure?
    """
    raise NotImplementedError()


def calc_curr_dens(grid: Grid1D3V, electrons: Particles, ions: Particles):
    # Current density via CIC for all three velocity components
    grid.J.fill(0)
    np.add.at(grid.J, electrons.idx.flatten(), electrons.v * electrons.q * (1 - electrons.cic_weights))
    np.add.at(grid.J, (electrons.idx.flatten() + 1) % grid.n_cells, electrons.v * electrons.q * electrons.cic_weights)
    np.add.at(grid.J, ions.idx.flatten(), ions.v * ions.q * (1 - ions.cic_weights))
    np.add.at(grid.J, (ions.idx.flatten() + 1) % grid.n_cells, ions.v * ions.q * ions.cic_weights)


def calc_fields_1D3V(grid: Grid1D3V, dt, bc):
    # Ey, Ez, By, Bz are calculated using Runge-Kutta 2 and Upwind second order
    # some undesirable properties: we do not check for charge conservation for these fields
    # desirable properties: charge conservation is implemented for Ex, that is why we calculate it using poisson solver at every timestep
    # a way we might be able to track the accuracy would be by tracking charge conservation and seeing if it holds well enough for our purposes as is.
    # if so, no further changes are needed.
    """
    #second order interpolation to determine J at full timesteps to second order accuracy
    B_half = np.empty_like(grid.B)
    E_half = np.empty_like(grid.E)


    if bc is BoundaryCondition.Periodic:
        #calculate the fields at half timesteps
        B_half[:,1] = grid.B[:,1] - (dt / 2) / (2 * grid.dx) * (3 * grid.E[:,2] - 4 * np.roll(grid.E, -1)[:,2] + np.roll(grid.E, -2)[:,2])
        B_half[:,2] = grid.B[:,2] - (dt / 2) / (2 * grid.dx) * (3 * grid.E[:,1] - 4 * np.roll(grid.E, 1)[:,1] + np.roll(grid.E, 2)[:,1])
        E_half[:,1] = grid.E[:,1] - (dt / 2)  * (grid.J[:,1] / eps_0 +  c*c / (2 * grid.dx) * (3 * grid.B[:,2] - 4 * np.roll(grid.B, 1)[:,2] + np.roll(grid.B, 2)[:,2]))
        E_half[:,2] = grid.E[:,2] - (dt / 2)  * (grid.J[:,2] / eps_0 +  c*c / (2 * grid.dx) * (3 * grid.B[:,1] - 4 * np.roll(grid.B, -1)[:,1] + np.roll(grid.B, -2)[:,1]))

        #calculate the fields at the full timestep
        grid.B[:,1] += - dt / (2 * grid.dx) * (3 * E_half[:,2] - 4 * np.roll(E_half, -1)[:,2] + np.roll(E_half, -2)[:,2])
        grid.B[:,2] += - dt / (2 * grid.dx) * (3 * E_half[:,1] - 4 * np.roll(E_half, 1)[:,1] + np.roll(E_half, 2)[:,1])
        grid.E[:,1] += - dt  * (grid.J[:,1] / eps_0 +  c*c / (2 * grid.dx) * (3 * B_half[:,2] - 4 * np.roll(B_half, 1)[:,2] + np.roll(B_half, 2)[:,2]))
        grid.E[:,2] += - dt  * (grid.J[:,2] / eps_0 +  c*c / (2 * grid.dx) * (3 * B_half[:,1] - 4 * np.roll(B_half, -1)[:,1] + np.roll(B_half, -2)[:,1]))
    else:
        #calculate the fields at half timesteps
        B_half[:-2,1] = grid.B[:-2,1] - (dt / 2) / (2 * grid.dx) * (3 * grid.E[:-2,2] - 4 * grid.E[1:-1,2] + grid.E[2:,2])
        B_half[2:,2] = grid.B[2:,2] - (dt / 2) / (2 * grid.dx) * (3 * grid.E[2:,1] - 4 * grid.E[1:-1,1] + grid.E[:-2,1])
        E_half[2:,1] = grid.E[2:,1] - (dt / 2)  * (grid.J[2:,1] / eps_0 +  c*c / (2 * grid.dx) * (3 * grid.B[2:,2] - 4 * grid.B[1:-1,2] + grid.B[:-2,2]))
        E_half[:-2,2] = grid.E[:-2,2] - (dt / 2)  * (grid.J[:-2,2] / eps_0 +  c*c / (2 * grid.dx) * (3 * grid.B[:-2,1] - 4 * grid.B[1:-1,1] + grid.B[2:,1]))

        #calculate the boundary values using interpolation
        B_half[-2,1] = 3 * B_half[-3,1] - 3 * B_half[-4,1] + B_half[-5,1]
        B_half[-1,1] = 3 * B_half[-2,1] - 3 * B_half[-3,1] + B_half[-4,1]
        B_half[1,2] = 3 * B_half[2,2] - 3 * B_half[3,2] + B_half[4,2]
        B_half[0,2] = 3 * B_half[1,2] - 3 * B_half[2,2] + B_half[3,2]
        E_half[1,1] = 3 * E_half[2,1] - 3 * E_half[3,1] + E_half[4,1]
        E_half[0,1] = 3 * E_half[1,1] - 3 * E_half[2,1] + E_half[3,1]
        E_half[-2,2] = 3 * E_half[-3,2] - 3 * E_half[-4,2] + E_half[-5,2]
        E_half[-1,2] = 3 * E_half[-2,2] - 3 * E_half[-3,2] + E_half[-4,2]

        #calculate the fields at the full timestep
        grid.B[:-2,1] += - dt / (2 * grid.dx) * (3 * E_half[:-2,2] - 4 * E_half[1:-1,2] + E_half[2:,2])
        grid.B[2:,2] += - dt / (2 * grid.dx) * (3 * E_half[2:,1] - 4 * E_half[1:-1,1] + E_half[:-2,1])
        grid.E[2:,1] += - dt  * (grid.J[2:,1] / eps_0 +  c*c / (2 * grid.dx) * (3 * B_half[2:,2] - 4 * B_half[1:-1,2] + B_half[:-2,2]))
        grid.E[:-2,2] += - dt  * (grid.J[:-2,2] / eps_0 +  c*c / (2 * grid.dx) * (3 * B_half[:-2,1] - 4 * B_half[1:-1,1] + B_half[2:,1]))

        #calculate the boundary values using interpolation
        grid.B[-2,1] = 3 * grid.B[-3,1] - 3 * grid.B[-4,1] + grid.B[-5,1]
        grid.B[-1,1] = 3 * grid.B[-2,1] - 3 * grid.B[-3,1] + grid.B[-4,1]
        grid.B[1,2] = 3 * grid.B[2,2] - 3 * grid.B[3,2] + grid.B[4,2]
        grid.B[0,2] = 3 * grid.B[1,2] - 3 * grid.B[2,2] + grid.B[3,2]
        grid.E[1,1] = 3 * grid.E[2,1] - 3 * grid.E[3,1] + grid.E[4,1]
        grid.E[0,1] = 3 * grid.E[1,1] - 3 * grid.E[2,1] + grid.E[3,1]
        grid.E[-2,2] = 3 * grid.E[-3,2] - 3 * grid.E[-4,2] + grid.E[-5,2]
        grid.E[-1,2] = 3 * grid.E[-2,2] - 3 * grid.E[-3,2] + grid.E[-4,2]
    """
    # Solve Euler's equation to find E_x
    euler_solver_1D3V(grid, dt, bc)

    B_temp = grid.B.copy()
    E_temp = grid.E.copy()

    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        B_temp[:, 1] += dt / grid.dx * (np.roll(grid.E, -1)[:, 2] - grid.E[:, 2])
        B_temp[:, 2] += dt / grid.dx * (np.roll(grid.E, 1)[:, 1] - grid.E[:, 1])
        E_temp[:, 1] += dt * (-grid.J[:, 1] / eps_0 + c * c / grid.dx * (np.roll(grid.B, 1)[:, 2] - grid.B[:, 2]))
        E_temp[:, 2] += dt * (-grid.J[:, 2] / eps_0 + c * c / grid.dx * (np.roll(grid.B, -1)[:, 1] - grid.B[:, 1]))
    else:
        # calculate the fields at the full timestep
        B_temp[:-1, 1] += dt / grid.dx * (grid.E[1:, 2] - grid.E[:-1, 2])
        B_temp[1:, 2] += dt / grid.dx * (grid.E[:-1, 1] - grid.E[1:, 1])
        E_temp[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 + c * c / grid.dx * (grid.B[:-1, 2] - grid.B[1:, 2]))
        E_temp[:-1, 2] += dt * (-grid.J[:-1, 2] / eps_0 + c * c / grid.dx * (grid.B[1:, 1] - grid.B[:-1, 1]))

        # calculate the boundary values using interpolation
        B_temp[-1, 1] = 3 * B_temp[-2, 1] - 3 * B_temp[-3, 1] + B_temp[-4, 1]
        B_temp[0, 2] = 3 * B_temp[1, 2] - 3 * B_temp[2, 2] + B_temp[3, 2]
        E_temp[0, 1] = 3 * E_temp[1, 1] - 3 * E_temp[2, 1] + E_temp[3, 1]
        E_temp[-1, 2] = 3 * E_temp[-2, 2] - 3 * E_temp[-3, 2] + E_temp[-4, 2]

    # Only replace y and z components of the fields
    # Ex has already been calculated
    grid.E[:, 1:] = E_temp[:, 1:]
    # Bx is constant
    grid.B[:, 1:] = B_temp[:, 1:]


def euler_solver_1D3V(grid: Grid1D3V, dt: float, bc: BoundaryCondition):
    grid.E[:, 0] += grid.dx * grid.rho / eps_0
    raise NotImplementedError()


# -----------------------------------------------------
# Trying to fix SOR. It works if you give it enough iterations.
# The problem with this is that your code gets way too slow.
# @jit makes it faster. Convergence is never reached withing 1,000 iterations
# no matter which definition of convergence you use.
# Timings for 10,000 iterations are ~ 1s, which would be the cost of one
# iteration of the main loop => infeasible...
# -----------------------------------------------------


@jit
def sor_solver(u, f, dx, max_iter=10000, tol=1e-6, omega=1.5):
    num_cells = np.size(f)
    norm_f = norm(f)
    dx2 = dx**2
    # Dividing is a costly operation which is performed max_iter * num_cells times.
    # Replacing it with a multiplication saves a considerable amount of time
    dx2_inv = 1 / dx**2
    for _ in range(max_iter):
        # E = Σ_i e_i^2
        total_error = 0
        for i in range(num_cells - 1):
            # Calculate (approximation of) absolute error ∆u(x) - f(x) ≈ 0
            residual = (u[i - 1] - 2 * u[i] + u[i + 1]) * dx2_inv - f[i]
            u[i] += omega * dx2 / 2 * residual
            total_error += residual**2
        residual = (u[-2] - 2 * u[-1] + u[0]) * dx2_inv - f[-1]
        u[-1] += omega * dx2 / 2 * residual
        total_error += residual**2
        if np.sqrt(total_error) < tol * norm_f:
            print("SOR Converged before maximum iterations!! I AM HAPPY!!")
            break


@jit
def sor_solver2(u, f, dx, max_iter=10000, tol=1e-6, omega=1.5):
    num_cells = np.size(f)
    for _ in range(max_iter):
        max_diff = 0
        for i in range(num_cells - 1):  # Inner grid points
            old_u = u[i]
            u[i] = (1 - omega) * u[i] + (omega / 2) * (u[i + 1] + u[i - 1] - dx**2 * f[i])
            max_diff = max(max_diff, abs(u[i] - old_u))
        u[-1] = (1 - omega) * u[-1] + (omega / 2) * (u[0] + u[-2] - dx**2 * f[-1])
        if max_diff < tol:
            print("SOR Converged before maximum iterations!! I AM HAPPY!!")
            break


if __name__ == "__main__":
    from numpy.linalg import norm
    import matplotlib.pyplot as plt
    import time

    N = 10000
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # u = np.zeros(N)
    u = 0.02 * np.sin(20 * x) - np.sin(x)
    # plt.plot(x, u)
    # plt.show()
    f = np.sin(x)
    dx = 2 * np.pi / N
    t_start = time.time()
    sor_solver2(u, f, dx, omega=1.5, max_iter=100000)
    t_end = time.time()
    print(f"Execution took {1000 * (t_end-t_start)} milliseconds")
    plt.plot(x, u)
    plt.plot(x, -np.sin(x), "--")
    plt.show()

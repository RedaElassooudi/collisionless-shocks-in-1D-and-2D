# Contains Maxwell solver / Poisson solver
# Solving Poisson's equation:
#   - https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#   - https://en.wikipedia.org/wiki/Relaxation_(iterative_method)
#   - https://en.wikipedia.org/wiki/Successive_over-relaxation
import numpy as np

from grids import Grid1D, Grid1D3V, Grid2D
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


# @jit
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


def calc_curr_dens_1D3V(grid: Grid1D3V, electrons: Particles, ions: Particles):
    # Current density via CIC for all three velocity components
    grid.J.fill(0)
    np.add.at(grid.J, electrons.idx.flatten(), electrons.v * electrons.q * (1 - electrons.cic_weights))
    np.add.at(grid.J, (electrons.idx.flatten() + 1) % grid.n_cells, electrons.v * electrons.q * electrons.cic_weights)
    np.add.at(grid.J, ions.idx.flatten(), ions.v * ions.q * (1 - ions.cic_weights))
    np.add.at(grid.J, (ions.idx.flatten() + 1) % grid.n_cells, ions.v * ions.q * ions.cic_weights)


def calc_fields_1D3V(grid: Grid1D3V, dt, bc):
    """
    Maxwell's equations for 1D3V become:

    dE_x/dt = -J_x / eps_0\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    dB_x/dt = 0\n
    dB_y/dt = dE_z/dx\n
    dB_z/dt = -dE_y/dx\n
    """
    # To discretize th efields we assume E is known at integer gridpoints while B is known at staggered gridpoints
    # The grid of B is defined to be the grid of E plus dx/2
    # The derivatives are then calculated via central difference
    # Store temp values of fields
    E = grid.E.copy()
    B = grid.B.copy()

    if bc is BoundaryCondition.Periodic:
        # Calculate the fields at the full timestep
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[:, 1] += dt * (-grid.J[:, 1] / eps_0 - 2 * c * c / grid.dx * (B[:, 2] - np.roll(B[:, 2], 1)))
        grid.E[:, 2] += dt * (-grid.J[:, 2] / eps_0 + 2 * c * c / grid.dx * (B[:, 1] - np.roll(B[:, 1], 1)))
        grid.B[:, 1] += dt * 2 / grid.dx * (np.roll(E[:, 2], -1) - E[:, 2])
        grid.B[:, 2] -= dt * 2 / grid.dx * (np.roll(E[:, 1], -1) - E[:, 1])
    else:
        # Calculate the fields at the full timestep
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 - 2 * c * c / grid.dx * (B[1:, 2] - B[:-1, 2]))
        grid.E[1:, 2] += dt * (-grid.J[1:, 2] / eps_0 + 2 * c * c / grid.dx * (B[1:, 1] - B[:-1, 1]))
        grid.B[:-1, 1] += dt * 2 / grid.dx * (E[1:, 2] - E[:-1, 2])
        grid.B[:-1, 2] -= dt * 2 / grid.dx * (E[1:, 1] - E[:-1, 1])

        # Calculate the boundary values using interpolation
        grid.E[0, 1] = 3 * grid.E[1, 1] - 3 * grid.E[2, 1] + grid.E[3, 1]
        grid.E[0, 2] = 3 * grid.E[1, 2] - 3 * grid.E[2, 2] + grid.E[3, 2]
        grid.B[-1, 1] = 3 * grid.B[-2, 1] - 3 * grid.B[-3, 1] + grid.B[-4, 1]
        grid.B[-1, 2] = 3 * grid.B[-2, 2] - 3 * grid.B[-3, 2] + grid.B[-4, 2]


def calc_E_1D3V(grid: Grid1D3V, dt, bc):
    """
    Maxwell's equations for 1D3V become:

    dE_x/dt = -J_x / eps_0\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    dB_x/dt = 0\n
    dB_y/dt = dE_z/dx\n
    dB_z/dt = -dE_y/dx\n
    """
    # The fields E and B are assumed to be known at the same gridpoints, we use upwind or downwind depending on the sign of the spatial derivative
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[:, 1] += dt * (-grid.J[:, 1] / eps_0 + c * c / grid.dx * (np.roll(grid.B[:, 2], 1) - grid.B[:, 2]))
        grid.E[:, 2] += dt * (-grid.J[:, 2] / eps_0 + c * c / grid.dx * (np.roll(grid.B[:, 1], -1) - grid.B[:, 1]))
    else:
        # calculate the fields at the full timestep
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 + c * c / grid.dx * (grid.B[:-1, 2] - grid.B[1:, 2]))
        grid.E[:-1, 2] += dt * (-grid.J[:-1, 2] / eps_0 + c * c / grid.dx * (grid.B[1:, 1] - grid.B[:-1, 1]))

        # calculate the boundary values using interpolation
        grid.E[0, 1] = 3 * grid.E[1, 1] - 3 * grid.E[2, 1] + grid.E[3, 1]
        grid.E[-1, 2] = 3 * grid.E[-2, 2] - 3 * grid.E[-3, 2] + grid.E[-4, 2]


def calc_B_1D3V(grid: Grid1D3V, dt, bc):
    """
    Maxwell's equations for 1D3V become:

    dE_x/dt = -J_x / eps_0\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    dB_x/dt = 0\n
    dB_y/dt = dE_z/dx\n
    dB_z/dt = -dE_y/dx\n
    """
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.B[:, 1] += dt / grid.dx * (np.roll(grid.E[:, 2], -1) - grid.E[:, 2])
        grid.B[:, 2] += dt / grid.dx * (np.roll(grid.E[:, 1], 1) - grid.E[:, 1])
    else:
        # calculate the fields at the full timestep
        grid.B[:-1, 1] += dt / grid.dx * (grid.E[1:, 2] - grid.E[:-1, 2])
        grid.B[1:, 2] += dt / grid.dx * (grid.E[:-1, 1] - grid.E[1:, 1])
        # calculate the boundary values using interpolation
        grid.B[-1, 1] = 3 * grid.B[-2, 1] - 3 * grid.B[-3, 1] + grid.B[-4, 1]
        grid.B[0, 2] = 3 * grid.B[1, 2] - 3 * grid.B[2, 2] + grid.B[3, 2]


def euler_solver_1D3V(grid: Grid1D3V, dt: float, bc: BoundaryCondition):
    if bc is BoundaryCondition.Periodic:
        # Solve via FFT
        # Remove mean field, physically required for periodic boundary conditions as quasi neutrality must be maintained
        grid.rho -= np.mean(grid.rho)
        rho_k = np.fft.fft(grid.rho)
        k = 2 * np.pi * np.fft.fftfreq(grid.n_cells) / grid.x_max  # k = 2 * pi * n / L (n = -N/2, -N/2+1, ..., N/2-1)

        # Avoid dividing by k = zero
        E_k = np.zeros_like(rho_k)
        nonzero = k != 0
        E_k[nonzero] = rho_k[nonzero] / (1j * k[nonzero] * eps_0)
        grid.E[:, 0] = np.fft.ifft(E_k)
    elif bc is BoundaryCondition.Absorbing:
        # Solve using first order implicit discretization assuming E(x_0) = 0 (excluding any external fields)
        grid.E[0, 0] = 0
        for i in range(1, grid.n_cells):
            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
        grid.E[:, 0] += grid.E_0[:, 0]
    elif bc is BoundaryCondition.Open:
        # Solve using first order implicit discretization
        grid.E[0, 0] = 0
        for i in range(1, grid.n_cells):
            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
        # To remove any effects by setting E(x_0) = 0 we redo the calcs for E in opposite direction using E(x_N) as our starting point
        # --> Not certain if this implementation is fully correct
        for i in range(grid.n_cells - 2, -1, -1):
            grid.E[i, 0] = grid[i + 1, 0] - grid.dx / eps_0 * grid.rho[i]
        grid.E[:, 0] += grid.E_0[:, 0]


# -----------------------------------------------------
# Trying to fix SOR. It works if you give it enough iterations.
# The problem with this is that your code gets way too slow.
# @jit makes it faster. Convergence is never reached withing 1,000 iterations
# no matter which definition of convergence you use.
# Timings for 10,000 iterations are ~ 1s, which would be the cost of one
# iteration of the main loop => infeasible...
# -----------------------------------------------------


# @jit
def sor_solver(u, f, dx, max_iter=10000, tol=1e-6, omega=1.5):
    num_cells = np.size(f)
    norm_f = np.norm(f)
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


# @jit
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


def calc_curr_dens_2D(grid: Grid1D3V, electrons: Particles, ions: Particles):
    # Current density via CIC for both velocity components
    grid.J.fill(0)
    # TODO: We're assuming periodic BC here, take into account params.bc!
    # Create array to get the correct index for adjacent points
    x_adj = np.zeros((electrons.N, 2), dtype=int)
    y_adj = np.zeros((electrons.N, 2), dtype=int)
    x_adj[:, 0] = 1
    y_adj[:, 1] = 1
    np.add.at(
        grid.J,
        (electrons.idx[:, 0], electrons.idx[:, 1]),
        electrons.v * electrons.q * (1 - electrons.cic_weights[:, :1]) * (1 - electrons.cic_weights[:, 1:]),
    )
    coord = (electrons.idx + x_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, :1] * (1 - electrons.cic_weights[:, 1:]))
    coord = (electrons.idx + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, 1:] * (1 - electrons.cic_weights[:, :1]))
    coord = (electrons.idx + x_adj + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, :1] * electrons.cic_weights[:, 1:])

    x_adj = np.zeros((ions.N, 2), dtype=int)
    y_adj = np.zeros((ions.N, 2), dtype=int)
    x_adj[:, 0] = 1
    y_adj[:, 1] = 1
    np.add.at(grid.J, (ions.idx[:, 0], ions.idx[:, 1]), ions.v * ions.q * (1 - ions.cic_weights[:, :1]) * (1 - ions.cic_weights[:, 1:]))
    coord = (ions.idx + x_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, :1] * (1 - ions.cic_weights[:, 1:]))
    coord = (ions.idx + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, 1:] * (1 - ions.cic_weights[:, :1]))
    coord = (ions.idx + x_adj + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, :1] * ions.cic_weights[:, 1:])


def calc_E_2D(grid: Grid2D, dt, bc):
    """
    Maxwell's equations for 2D become:

    dE_x/dt = -J_x / eps_0 + c² dB_z/dy\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dB_z/dt = dE_x/dy - dE_y/dx\n
    """
    # Solve Euler's equation to find E
    # euler_solver_2D(grid, dt, bc)

    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.E[:, :, 0] += dt * (-grid.J[:, :, 0] / eps_0 + c * c / grid.dx * (grid.B[:, :, 0] - np.roll(grid.B[:, :, 0], 1, axis=0)))
        grid.E[:, :, 1] += dt * (-grid.J[:, :, 1] / eps_0 - c * c / grid.dx * (grid.B[:, :, 0] - np.roll(grid.B[:, :, 0], 1, axis=1)))
    else:
        # calculate the fields at the full timestep
        grid.E[1:, :, 0] += dt * (-grid.J[1:, :, 0] / eps_0 + c * c / grid.dx * (grid.B[1:, :, 0] - np.roll(grid.B[:-1, :, 0], 1, axis=0)))
        grid.E[:, 1:, 1] += dt * (-grid.J[:, 1:, 1] / eps_0 - c * c / grid.dx * (grid.B[:, 1:, 0] - np.roll(grid.B[:, :-1, 0], 1, axis=1)))

        # calculate the boundary values using interpolation
        grid.E[0, :, 0] = 3 * grid.E[1, :, 0] - 3 * grid.E[2, :, 0] + grid.E[3, :, 0]
        grid.E[0, :, 1] = 3 * grid.E[:, 1, 1] - 3 * grid.E[:, 2, 1] + grid.E[:, 3, 1]


def calc_B_2D(grid: Grid2D, dt, bc):
    """
    dB_z/dt = dE_x/dy - dE_y/dx\n
    """
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.B[:, :, 0] += (
            dt
            * c
            * c
            * (
                -(np.roll(grid.E[:, :, 1], -1, axis=1) - grid.E[:, :, 1]) / grid.dx
                + (np.roll(grid.E[:, :, 0], -1, axis=0) - grid.E[:, :, 0]) / grid.dx
            )
        )
    else:
        # calculate the fields at the full timestep
        grid.B[:-1, :-1, 0] += (
            dt * c * c * (-(grid.E[1:, 1:, 1] - grid.E[:-1, :-1, 1]) / grid.dx + (grid.E[1:, 1:, 0] - grid.E[:-1, :-1, 0]) / grid.dx)
        )

        # calculate the boundary values using interpolation
        grid.B[-1, :-1, 0] = 3 * grid.B[-2, :-1, 0] - 3 * grid.B[-3, :-1, 0] + grid.B[-4, :-1, 0]
        grid.B[:-1, -1, 0] = 3 * grid.B[:-1, -2, 0] - 3 * grid.B[:-1, -3, 0] + grid.B[:-1, -4, 0]
        grid.B[-1, -1, 0] = 3 * grid.B[-1, -2, 0] - 3 * grid.B[-1, -3, 0] + grid.B[-1, -4, 0]  # last value is an interpolation of an interpolation


"""
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
"""

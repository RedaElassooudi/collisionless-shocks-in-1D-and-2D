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
    # E_i = - (phi_i+1 - phi_i) / dx
    grid.E[:-1] = -(grid.phi[1:] - grid.phi[:-1]) / params.dx
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


# -----------------------------------------------------
# Comments while trying to fix SOR: It works if you give it enough iterations.
# The problem with this is that your code gets way too slow.
# @jit makes it faster. Convergence is never reached within 1,000 iterations
# no matter which definition of convergence you use (max update / total error norm).
# Timings for 10,000 iterations are ~ 1s, which would be the cost of one
# iteration of the main loop => infeasible...
# -----------------------------------------------------
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


def calc_fields_1D3V(grid: Grid1D3V, dt):
    """
    Solve Maxwell's equations on Yee grid **assuming PERIODIC boundary conditions**!!!\n
    Uses sub-cycling, and limiter on B\n
    Maxwell's equations for 1D3V become:

    * dE_x/dt = -J_x / eps_0\n
    * dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    * dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    * dB_x/dt = 0\n
    * dB_y/dt = dE_z/dx\n
    * dB_z/dt = -dE_y/dx\n
    """
    # To discretize the fields we assume E is known at integer gridpoints while B is known at staggered gridpoints
    # The grid of B is defined to be the grid of E plus dx/2
    # The derivatives are then calculated via central difference
    # Store temp values of fields
    E = grid.E.copy()
    dt /= 2

    # Update E unrestricted for +dt/2
    grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
    grid.E[:, 1] += dt * (-grid.J[:, 1] / eps_0 - c * c / grid.dx * (grid.B[:, 2] - np.roll(grid.B[:, 2], 1)))
    grid.E[:, 2] += dt * (-grid.J[:, 2] / eps_0 + c * c / grid.dx * (grid.B[:, 1] - np.roll(grid.B[:, 1], 1)))

    # Update B restricted by limiter for +dt/2
    ## Calculate update
    dBy = dt / grid.dx * (np.roll(E[:, 2], -1) - E[:, 2])
    dBz = -dt / grid.dx * (np.roll(E[:, 1], -1) - E[:, 1])
    ## Apply limiter
    # max_change = 0.1  # Maximum allowed relative change
    # dBy = np.clip(dBy, -max_change * np.abs(grid.B[:, 1]) - 1e-5, max_change * np.abs(grid.B[:, 1]) + 1e-5)
    # dBz = np.clip(dBz, -max_change * np.abs(grid.B[:, 2]) - 1e-5, max_change * np.abs(grid.B[:, 2]) + 1e-5)
    ## Apply update
    grid.B[:, 1] += dBy
    grid.B[:, 2] += dBz

    # Remember B at t = n + 1/2 for update to E later
    B = grid.B.copy()

    # Update B restricted by limiter for +dt/2 (total: +dt)
    dBy = dt / grid.dx * (np.roll(grid.E[:, 2], -1) - grid.E[:, 2])
    dBz = -dt / grid.dx * (np.roll(grid.E[:, 1], -1) - grid.E[:, 1])
    # dBy = np.clip(dBy, -max_change * np.abs(B[:, 1]) - 1e-5, max_change * np.abs(B[:, 1]) + 1e-5)
    # dBz = np.clip(dBz, -max_change * np.abs(B[:, 2]) - 1e-5, max_change * np.abs(B[:, 2]) + 1e-5)
    grid.B[:, 1] += dBy
    grid.B[:, 2] += dBz

    # Update E unrestricted for +dt/2 (total: +dt)
    grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
    grid.E[:, 1] += dt * (-grid.J[:, 1] / eps_0 - c * c / grid.dx * (B[:, 2] - np.roll(B[:, 2], 1)))
    grid.E[:, 2] += dt * (-grid.J[:, 2] / eps_0 + c * c / grid.dx * (B[:, 1] - np.roll(B[:, 1], 1)))


def calc_fields_1D3V_nonperiodic(grid: Grid1D3V, dt):
    E = grid.E.copy()
    max_change = 0.1  # Maximum allowed relative change
    # Calculate the fields at the full timestep
    grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
    grid.E[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 - c * c / grid.dx * (grid.B[1:, 2] - grid.B[:-1, 2]))
    grid.E[1:, 2] += dt * (-grid.J[1:, 2] / eps_0 + c * c / grid.dx * (grid.B[1:, 1] - grid.B[:-1, 1]))
    dBy = dt / grid.dx * (E[1:, 2] - E[:-1, 2])
    dBz = -dt / grid.dx * (E[1:, 1] - E[:-1, 1])

    # Apply limiters to interior points
    dBy = np.clip(dBy, -max_change * np.abs(grid.B[:-1, 1]), max_change * np.abs(grid.B[:-1, 1]))
    dBz = np.clip(dBz, -max_change * np.abs(grid.B[:-1, 2]), max_change * np.abs(grid.B[:-1, 2]))

    grid.B[:-1, 1] += dBy
    grid.B[:-1, 2] += dBz

    # Calculate the boundary values using interpolation
    grid.E[0, 1] = 3 * grid.E[1, 1] - 3 * grid.E[2, 1] + grid.E[3, 1]
    grid.E[0, 2] = 3 * grid.E[1, 2] - 3 * grid.E[2, 2] + grid.E[3, 2]
    By_right = 3 * grid.B[-2, 1] - 3 * grid.B[-3, 1] + grid.B[-4, 1]
    Bz_right = 3 * grid.B[-2, 2] - 3 * grid.B[-3, 2] + grid.B[-4, 2]

    # Apply limiters to boundary values
    dBy_right = By_right - grid.B[-1, 1]
    dBz_right = Bz_right - grid.B[0, 2]

    dBy_right = np.clip(dBy_right, -max_change * np.abs(grid.B[-1, 1]), max_change * np.abs(grid.B[-1, 1]))
    dBz_right = np.clip(dBz_right, -max_change * np.abs(grid.B[-1, 2]), max_change * np.abs(grid.B[-1, 2]))

    grid.B[-1, 1] += dBy_right
    grid.B[0, 2] += dBz_right


def calc_E_1D3V(grid: Grid1D3V, dt, bc):
    """
    dE_x/dt = -J_x / eps_0\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
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


# def euler_solver_1D3V(grid: Grid1D3V, dt: float, bc: BoundaryCondition):
#     if bc is BoundaryCondition.Periodic:
#        # Solve via FFT
#        # Remove mean field, physically required for periodic boundary conditions as quasi neutrality must be maintained
#        grid.rho -= np.mean(grid.rho)
#        rho_k = np.fft.fft(grid.rho)
#        k = 2 * np.pi * np.fft.fftfreq(grid.n_cells) / grid.x_max  # k = 2 * pi * n / L (n = -N/2, -N/2+1, ..., N/2-1)
#
#        # Avoid dividing by k = zero
#        E_k = np.zeros_like(rho_k)
#        nonzero = k != 0
#        E_k[nonzero] = rho_k[nonzero] / (1j * k[nonzero] * eps_0)
#        grid.E[:, 0] = np.fft.ifft(E_k)
#    elif bc is BoundaryCondition.Absorbing:
#        # Solve using first order implicit discretization assuming E(x_0) = 0 (excluding any external fields)
#        grid.E[0, 0] = 0
#        for i in range(1, grid.n_cells):
#            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
#        grid.E[:, 0] += grid.E_0[:, 0]
#    elif bc is BoundaryCondition.Open:
#        # Solve using first order implicit discretization
#        grid.E[0, 0] = 0
#        for i in range(1, grid.n_cells):
#            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
#        # To remove any effects by setting E(x_0) = 0 we redo the calcs for E in opposite direction using E(x_N) as our starting point
#        # --> Not certain if this implementation is fully correct
#        for i in range(grid.n_cells - 2, -1, -1):
#            grid.E[i, 0] = grid[i + 1, 0] - grid.dx / eps_0 * grid.rho[i]
#        grid.E[:, 0] += grid.E_0[:, 0]


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

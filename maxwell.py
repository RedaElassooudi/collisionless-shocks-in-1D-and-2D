# Contains Maxwell solver / Poisson solver
# Solving Poisson's equation:
#   - https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#   - https://en.wikipedia.org/wiki/Relaxation_(iterative_method)
#   - https://en.wikipedia.org/wiki/Successive_over-relaxation
import numpy as np

from physical_constants import *
from grids import Grid1D, Grid1D3V
from parameters import BoundaryCondition, Parameters
from particles import Particles


def poisson_solver_1D(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, first=False):
    grid.set_densities(electrons, ions)
    # If this is the first time we solve the poisson equations,
    # we need a good initial guess for the iterative SOR solver,
    # else convergence can't be achieved within a reasonable number of iterations.
    # TODO: commented it out so that we can have a result. The SOR solver is very very slow, it never converges.
    # 1000 SOR iterations / outer iteration really add up (computation time blows up due to this)
    # if first:
    #     naive_poisson_solver(grid, params.dx)
    # solve_poisson_sor(grid.phi, grid.rho, params.dx, params.bc, params.SOR_max_iter, params.SOR_tol, params.SOR_omega)
    naive_poisson_solver_1D(grid, params.dx)

    # Electric field calculation
    grid.E[:-1] = -(grid.phi[1:] - grid.phi[:-1]) / params.dx + grid.E_0[:-1]
    #take boundary conditions into account
    if params.bc is BoundaryCondition.Periodic: 
        # TODO: this assumes periodic boundary conditions, do not uncomment or you will get EXTREMELY large field at the right boundary
        #   The fix also has its issues
        # grid.E[-1] = -(grid.phi[0] - grid.phi[-1]) / params.dx + grid.E_0[-1]
        pass
    else: #use second order interpolation to get the last value
        grid.E[-1] = 2 * grid.E[-3] - grid.E[-2] + grid.E_0[-1]


def naive_poisson_solver_1D(grid: Grid1D, dx: float):
    grid.phi.fill(0)
    grid.phi[1:] = np.cumsum(np.cumsum(grid.rho[:-1])) * dx**2


def poisson_solver_1D3V(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, first=False):
    grid.set_densities(electrons, ions)
    # If this is the first time we solve the poisson equations,
    # we need a good initial guess for the iterative SOR solver,
    # else convergence can't be achieved within a reasonable number of iterations.
    # TODO: commented it out so that we can have a result. The SOR solver is very very slow, it never converges.
    # 1000 SOR iterations / outer iteration really add up (computation time blows up due to this)
    # if first:
    #     naive_poisson_solver(grid, params.dx)
    # solve_poisson_sor(grid.phi, grid.rho, params.dx, params.bc, params.SOR_max_iter, params.SOR_tol, params.SOR_omega)
    naive_poisson_solver_1D3V(grid, params.dx)

    # Electric field calculation
    grid.E[:-1,0] = -(grid.phi[1:] - grid.phi[:-1]) / params.dx + grid.E_0[:-1,0]
    #take boundary conditions into account
    if params.bc is BoundaryCondition.Periodic: 
        # TODO: this assumes periodic boundary conditions, do not uncomment or you will get EXTREMELY large field at the right boundary
        #  The fix also has its issues
        # grid.E[-1,0] = -(grid.phi[0] - grid.phi[-1]) / params.dx + grid.E_0[-1,0]
        pass
    else: #use second order interpolation to get the last value
        grid.E[-1,0] = 2 * grid.E[-3,0] - grid.E[-2,0] + grid.E_0[-1,0]


def naive_poisson_solver_1D3V(grid: Grid1D, dx: float):
    grid.phi.fill(0)
    grid.phi[1:] = np.cumsum(np.cumsum(grid.rho[:-1])) * dx**2



# TODO: if we keep using this solver, try numba
def solve_poisson_sor(
    phi, rho_tilde, dx, bound_cond, max_iter=1000, tol=1e-6, omega=1.5
):  # max_iter is number of iterations, tol is error tollerance, omega is relaxation factor
    """
    Update for the SOR process for the electric field
    """
    num_cells = len(rho_tilde)

    # choice of boundary conditions determines the range over which this solver is applied
    if bound_cond is BoundaryCondition.Open or bound_cond is BoundaryCondition.Absorbing:
        for _ in range(max_iter):
            max_error = 0
            for i in range(1, num_cells - 1):  # Inner grid points
                old_phi = phi[i]
                phi[i] = (1 - omega) * phi[i] + (omega / 2) * (phi[i + 1] + phi[i - 1] + dx**2 * rho_tilde[i])
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
                phi[i - 1] = (1 - omega) * phi[i - 1] + (omega / 2) * (phi[i] + phi[i - 2] + dx**2 * rho_tilde[i - 1])
                max_error = max(max_error, abs(phi[i - 1] - old_phi))
            if max_error < tol:
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


def calc_curr_dens(grid: Grid1D3V, electrons: Particles, ions: Particles):
    # Current density via CIC for all three velocity components
    grid.J.fill(0)
    np.add.at(grid.J, electrons.idx.flatten(), electrons.v * electrons.q * (1 - electrons.cic_weights))
    np.add.at(grid.J, (electrons.idx.flatten() + 1) % grid.n_cells, electrons.v * electrons.q * electrons.cic_weights)
    np.add.at(grid.J, ions.idx.flatten(), ions.v * ions.q * (1 - ions.cic_weights))
    np.add.at(grid.J, (ions.idx.flatten() + 1) % grid.n_cells, ions.v * ions.q * ions.cic_weights)

def calc_fields(grid: Grid1D3V, dt, bc):
    #Ey, Ez, By, Bz are calculated using Runge-Kutta 2 and Upwind second order 
    #some undesirable properties: we do not check for charge conservation for these fields
    #desirable properties: charge conservation is implemented for Ex, that is why we calculate it using poisson solver at every timestep
    #a way we might be able to track the accuracy would be by tracking charge conservation and seeing if it holds well enough for our purposes as is.
    #if so, no further changes are needed.

    #second order interpolation to determine J at full timesteps to second order accuracy
    B_half = np.empty_like(grid.B)
    E_half = np.empty_like(grid.E)
    
    """
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
    
    if bc is BoundaryCondition.Periodic:
        #calculate the fields at the full timestep
        grid.B[:,1] += dt / grid.dx * (np.roll(E_half, -1)[:,2] - E_half[:,2])
        grid.B[:,2] += dt / grid.dx * (np.roll(E_half, 1)[:,1] - E_half[:,1])
        grid.E[:,1] += dt  * (-grid.J[:,1] / eps_0 +  c*c / grid.dx * (np.roll(B_half, 1)[:,2] - B_half[:,2]))
        grid.E[:,2] += dt  * (-grid.J[:,2] / eps_0 +  c*c / grid.dx * (np.roll(B_half, -1)[:,1] - B_half[:,1]))
    else:
        #calculate the fields at the full timestep
        grid.B[:-1,1] += dt / grid.dx * (E_half[1:,2] - E_half[:-1,2])
        grid.B[1:,2] += dt / grid.dx * (E_half[:-1,1] - E_half[1:,1])
        grid.E[1:,1] += dt  * (-grid.J[1:,1] / eps_0 +  c*c / grid.dx * (B_half[:-1,2] - B_half[1:,2]))
        grid.E[:-1,2] += dt  * (-grid.J[:-1,2] / eps_0 +  c*c / grid.dx * (B_half[1:,1] - B_half[:-1,1]))

        #calculate the boundary values using interpolation
        grid.B[-1,1] = 3 * grid.B[-2,1] - 3 * grid.B[-3,1] + grid.B[-4,1]
        grid.B[0,2] = 3 * grid.B[1,2] - 3 * grid.B[2,2] + grid.B[3,2]
        grid.E[0,1] = 3 * grid.E[1,1] - 3 * grid.E[2,1] + grid.E[3,1]
        grid.E[-1,2] = 3 * grid.E[-2,2] - 3 * grid.E[-3,2] + grid.E[-4,2]
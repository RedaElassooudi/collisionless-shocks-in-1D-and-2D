import numpy as np


from initial_distributions import initialize_particles
from parameters import Parameters, BoundaryCondition
import solver_1D3V
import visualizations as vis

if __name__ == "__main__":

    np.random.seed(42)

    num_particles = 20000  # Total number of particles (ions + electrons)
    num_cells = 200  # Number of spatial grid cells
    dx = 1.0  # Spatial step size
    x_max = num_cells * dx  # Maximum position value
    t_max = 1.0e0
    damping_width = x_max // 10  # Size of region where dampening will occur

    # Shock generation parameters
    bulk_velocity_e = 2.0  # Bulk velocity for electrons (towards left)
    bulk_velocity_i = 0.2  # Bulk velocity for ions (towards left)

    el, io = initialize_particles(num_particles, x_max, bulk_velocity_e, bulk_velocity_i, 3)
    params = Parameters(x_max, dx, t_max, BoundaryCondition.Periodic, damping_width)
    res = solver_1D3V.simulate(el, io, params)

    # Define time steps to plot (start, middle, end)
    time_steps = [0, len(res.t) // 2, len(res.t) - 1]

    # Plot results
    # Plot Ex and By
    vis.field_ND(time_steps, res.x, res.E, 0, "Electric Field (Ex)", res.t)
    vis.field_ND(time_steps, res.x, res.B, 1, "Magnetic Field (By)", res.t)
    vis.density_profiles_1D(time_steps, res.x, res.n_e, res.n_i, res.t)
    vis.energy_evolution(res.t, res.KE, res.PE, res.TE)
    # Problem: where does v_i go?? Maybe spread too thin so very narrow bins?
    vis.velocity_profiles_ND(time_steps, res.v_e, res.v_i, res.t, 0)
    # vis.phase_space_1D(time_steps, res.x_e, res.v_e, res.x_i, res.v_i, res.t)
    # vis.animate_phase_space(res.x_e, res.v_e, x_max)

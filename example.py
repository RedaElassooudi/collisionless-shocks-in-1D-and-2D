import numpy as np


from initial_distributions import initialize_particles, two_stream
from parameters import Parameters, BoundaryCondition
import solver_2D
import visualizations as vis

if __name__ == "__main__":

    np.random.seed(42)

    num_particles = 20000  # Total number of particles (ions + electrons)
    num_cells = 100  # Number of spatial grid cells
    x_max = 1.0  # Maximum position value
    dx = x_max / num_cells  # Spatial step size
    dimx = 2
    dimv = 2

    t_max = 3.0e2
    max_iter = 20000
    damping_width = x_max // 10  # Size of region where dampening will occur

    # Shock generation parameters
    bulk_velocity_e = 5e-7  # Bulk velocity for electrons (towards left)
    bulk_velocity_i = 5e-7  # Bulk velocity for ions (towards left)
    
    el, io = initialize_particles(num_particles, x_max, bulk_velocity_e, bulk_velocity_i, dimx, dimv)
    """
    # Fabio code: instability occurs at w_pe * t = 25
    V0 = 0.9  # Stream velocity
    VT = 0.00000001  # Thermal speed
    el, io = two_stream(num_particles, x_max, VT, V0)
    """
    params = Parameters(x_max, dx, t_max, max_iter, BoundaryCondition.Periodic, damping_width, dimx, dimv)
    res = solver_2D.simulate(el, io, params)

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
    vis.phase_space_ND(time_steps, res.x_e, res.v_e, res.x_i, res.v_i, res.t)
    # vis.animate_phase_space(res.x_e, res.v_e, x_max)

    # You can put this code in a different file and it will still work
    # In fact, you are meant to put it in a different file.
    # You should run the code once, and then perform multiple inspections / visualizations
    # with code like below without having to recompute everything
    # (Make sure to change the directory name, as the number will be different)
    # The number represents the time in seconds since 00:00:00 UTC 1 Jan 1970
    # Make sure you use the correct names for the different variables (see results.py for their names)
    # KE = Results.read("Results/1734833451/", "KE")
    # PE = Results.read("Results/1734833451/", "PE")
    # TE = Results.read("Results/1734833451/", "TE")
    # t = Results.read("Results/1734833451/", "t")
    # vis.energy_evolution(t, KE, PE, TE)

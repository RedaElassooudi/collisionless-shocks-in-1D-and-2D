import numpy as np

import sys

sys.path.append("./")
sys.path.append("../")

from initial_distributions import initialize_particles, two_stream
from parameters import Parameters, BoundaryCondition
from results import Results
import solver_1D3V
import visualizations.visualizations as vis


def main():
    np.random.seed(42)

    num_particles = 20000  # Total number of particles (ions + electrons)
    num_cells = 100  # Number of spatial grid cells
    x_max = 1.0  # Maximum position value
    dx = x_max / num_cells  # Spatial step size
    dimx = 1
    dimv = 3

<<<<<<< Updated upstream
    t_max = 1.0e1
=======
    t_max = 5e0
>>>>>>> Stashed changes
    max_iter = 20000
    damping_width = x_max // 10  # Size of region where dampening will occur

    # Shock generation parameters
    bulk_velocity_e = 5e-7  # Bulk velocity for electrons (towards left)
    bulk_velocity_i = 5e-7  # Bulk velocity for ions (towards left)

    """
    el, io = initialize_particles(num_particles, x_max, bulk_velocity_e, bulk_velocity_i, dimx, dimv)
    """
    # Fabio code: instability occurs at w_pe * t = 25

    params = Parameters(x_max, dx, t_max, max_iter, BoundaryCondition.Periodic, dimX=dimx, dimV=dimv)

    # Fabio code: instability occurs at w_pe * t = 25
    v_bulk = 0.9  # Stream velocity
    v_th = 0.00000001  # Thermal speed
    pert_amp = 0.0001  # Perturbation amplitude
    mode = 3  # wave mode to activate
    el, io = two_stream(num_particles, x_max, v_th, v_bulk, num_cells, pert_amp, mode)

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
    vis.phase_space_ND(time_steps, res.x_e, res.v_e, res.x_i, res.v_i, res.t)
    # vis.animate_phase_space(res.x_e, res.v_e, x_max)


def read_and_vis():
    # You can put this code in a different file and it will still work
    # In fact, you are meant to put it in a different file.
    # You should run the code once, and then perform multiple inspections / visualizations
    # with code like below without having to recompute everything
    # (Make sure to change the directory name, as the number will be different)
    # The number represents the time in seconds since 00:00:00 UTC 1 Jan 1970
    # Make sure you use the correct names for the different variables (see results.py for their names)
    KE = Results.read("Results/1734862082/", "KE")
    PE = Results.read("Results/1734862082/", "PE")
    TE = Results.read("Results/1734862082/", "TE")
    t = Results.read("Results/1734862082/", "t")
    x = Results.read("Results/1734862082/", "x")
    n_e = Results.read("Results/1734862082/", "n_e")
    n_i = Results.read("Results/1734862082/", "n_i")
    print(min(TE))
    print(max(TE))

    # Define time steps to plot (start, middle, end)
    time_steps = [0, len(t) // 2, len(t) - 1]

    vis.density_profiles_1D(time_steps, x, n_e, n_i, t)
    vis.energy_evolution(t, KE, PE, TE)


if __name__ == "__main__":
    # read_and_vis()
    main()

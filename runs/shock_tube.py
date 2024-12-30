import sys

sys.path.append("./")
sys.path.append("../")

# Import your local modules for the simulation
from initial_distributions import shock_tube
from parameters import Parameters  # If it doesn't have boundary_condition/damping_width, remove them below.
from parameters import BoundaryCondition  # Only if your code has this. Otherwise remove it.
import solver_1D3V, solver_1D
import visualizations.visualizations2 as vis
import visualizations.shockplot as sp
import numpy as np


# Helper to compute density compression, shock thickness
def density_compression_ratio(x_ts, n_ts, x_shock, width=0.02):
    downstream_mask = (x_ts >= x_shock - 2 * width) & (x_ts < x_shock - width)
    upstream_mask = (x_ts > x_shock + width) & (x_ts <= x_shock + 2 * width)
    n_down = np.mean(n_ts[downstream_mask])
    n_up = np.mean(n_ts[upstream_mask])
    if n_up is None or n_up == 0 or np.isnan(n_up):
        return np.nan
    return n_down / n_up


def shock_thickness(x_ts, n_ts, n_up, frac_low=0.1, frac_high=0.9):
    n_peak = np.max(n_ts)
    low_val = n_up + frac_low * (n_peak - n_up)
    high_val = n_up + frac_high * (n_peak - n_up)
    indices_low = np.where(n_ts >= low_val)[0]
    indices_high = np.where(n_ts >= high_val)[0]
    if len(indices_low) == 0 or len(indices_high) == 0:
        return 0.0
    x_low = x_ts[indices_low[0]]
    x_high = x_ts[indices_high[0]]
    return x_high - x_low


if __name__ == "__main__":

    np.random.seed(42)

    num_particles = 50_000
    num_cells = 200
    x_max = 1.0
    dx = x_max / num_cells
    t_max = 2.5e0
    max_iter = 20_000

    V0 = 0.9
    VT = 1e-8

    el, io = shock_tube(num_particles, x_max, VT, V0)

    params = Parameters(x_max, num_cells, t_max, max_iter, BoundaryCondition.Periodic, 1, 3, num_particles=num_particles)

    res_3v = solver_1D3V.simulate(el, io, params)

    time_steps = [0, len(res_3v.t) // 2, len(res_3v.t) - 1]

    vis.field_ND(time_steps, res_3v.x, res_3v.E, 0, "Electric Field (Ex)", res_3v.t)
    vis.field_ND(time_steps, res_3v.x, res_3v.B, 1, "Magnetic Field (By)", res_3v.t)
    vis.density_profiles_1D(time_steps, res_3v.x, res_3v.n_e, res_3v.n_i, res_3v.t)
    vis.energy_evolution(res_3v.t, res_3v.KE, res_3v.PE, res_3v.TE)
    vis.velocity_profiles_ND(time_steps, res_3v.v_e, res_3v.v_i, res_3v.t, component=0)
    vis.phase_space_ND(time_steps, res_3v.x_e, res_3v.v_e, res_3v.x_i, res_3v.v_i, res_3v.t, component=0)

    # Shock analysis
    res_3v.x = np.array(res_3v.x)
    res_3v.n_i = np.array(res_3v.n_i)
    res_3v.n_e = np.array(res_3v.n_e)
    res_3v.E = np.array(res_3v.E)
    res_3v.B = np.array(res_3v.B)

    Nt = len(res_3v.t)
    Nx = res_3v.n_i.shape[1]

    shock_positions = np.zeros(Nt)
    shock_thickness_vs_time = np.zeros(Nt)
    compression_ratio_i = np.zeros(Nt)
    compression_ratio_e = np.zeros(Nt)

    for ts in range(Nt):
        x_ts = res_3v.x[ts]
        n_i_ts = res_3v.n_i[ts]
        n_e_ts = res_3v.n_e[ts]

        i_max = np.argmax(n_i_ts)
        x_shock = x_ts[i_max]
        shock_positions[ts] = x_shock
        n_up = np.mean(n_i_ts[:10])
        shock_thickness_vs_time[ts] = shock_thickness(x_ts, n_i_ts, n_up)
        compression_ratio_i[ts] = density_compression_ratio(x_ts, n_i_ts, x_shock)
        compression_ratio_e[ts] = density_compression_ratio(x_ts, n_e_ts, x_shock)

    # Shock speed & Mach
    shock_speed_vs_time = np.zeros(Nt)
    for ts in range(1, Nt):
        dt = res_3v.t[ts] - res_3v.t[ts - 1]
        shock_speed_vs_time[ts] = (shock_positions[ts] - shock_positions[ts - 1]) / dt
    c_s = 0.01
    mach_number = shock_speed_vs_time / c_s

    coefs = np.polyfit(res_3v.t, shock_positions, 1)
    shock_speed_fit = coefs[0]
    print("Average shock speed (linear fit):", shock_speed_fit)

    # 3) Compute density compression & shock thickness at final time
    final_ts = len(res_3v.t) - 1

    x_final = res_3v.x[final_ts]  # shape (Nx,)
    n_final_i = res_3v.n_i[final_ts]  # Ion density
    n_final_e = res_3v.n_e[final_ts]  # Electron density
    final_shock_pos = shock_positions[final_ts]

    # 3a) Ion compression ratio at final time
    final_ratio_ions = density_compression_ratio(x_ts=x_final, n_ts=n_final_i, x_shock=final_shock_pos, width=0.02)
    print("Final density compression ratio (ions):", final_ratio_ions)

    # 3b) Electron compression ratio at final time
    final_ratio_electrons = density_compression_ratio(x_ts=x_final, n_ts=n_final_e, x_shock=final_shock_pos, width=0.02)
    print("Final density compression ratio (electrons):", final_ratio_electrons)

    # 3c) Shock thickness at final time (using ions as reference)
    # For upstream density, let's average the first ~10 cells:
    n_up = np.mean(n_final_i[:10])
    thickness = shock_thickness(x_ts=x_final, n_ts=n_final_i, n_up=n_up)
    print("Shock thickness at final time:", thickness)

    shock_speed_vs_time = np.zeros(Nt)
    shock_speed_vs_time[0] = 0.0
    for ts in range(1, Nt):
        dt = res_3v.t[ts] - res_3v.t[ts - 1]
        shock_speed_vs_time[ts] = (shock_positions[ts] - shock_positions[ts - 1]) / dt

    # (4) Mach number
    c_s = 0.01  # example
    mach_number = shock_speed_vs_time / c_s

    # Plot line-based shock properties
    sp.plot_shock_position_time(res_3v.t, shock_positions)
    sp.plot_shock_thickness_time(res_3v.t, shock_thickness_vs_time)
    sp.plot_compression_ratio_time(res_3v.t, compression_ratio_i, compression_ratio_e)
    sp.plot_mach_number_time(res_3v.t, mach_number)

    # If energies exist:
    sp.plot_energy_time(res_3v.t, res_3v.KE, res_3v.PE, res_3v.TE)

    # 2D color maps
    sp.plot_space_time_density(res_3v.t, res_3v.x, res_3v.n_i, species="ion")
    sp.plot_space_time_density(res_3v.t, res_3v.x, res_3v.n_e, species="electron")
    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.E[..., 0], field_name="Ex")
    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.E[..., 1], field_name="Ey")
    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.E[..., 2], field_name="Ez")

    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.B[..., 0], field_name="Bx")
    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.B[..., 1], field_name="By")
    sp.plot_space_time_field(res_3v.t, res_3v.x, res_3v.B[..., 2], field_name="Bz")

    # sp.plot_field_spectrogram_in_space(res_3v.t, res_3v.E[..., 0], 1, field_name="Ex", k_max=None, log_scale=True)

    # Phase-space & velocity distributions
    mid_ts = Nt // 2
    sp.plot_phase_space_1D(mid_ts, res_3v.x_i, res_3v.v_i, time_val=res_3v.t[mid_ts])
    sp.plot_vx_vy_distribution(mid_ts, res_3v.v_e, species="electrons", nbins=60)

    print("Simulation and advanced shock plotting complete!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_shock_position_time(t, shock_position, label="Shock Position"):
    
    plt.figure()
    plt.plot(t, shock_position, marker='o', label=label)
    plt.xlabel("Time")
    plt.ylabel("Shock Position (x)")
    plt.title("Shock Position vs. Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_shock_thickness_time(t, shock_thickness, label="Shock Thickness"):
  
    plt.figure()
    plt.plot(t, shock_thickness, marker='o', label=label, color='red')
    plt.xlabel("Time")
    plt.ylabel("Shock Thickness (x-units)")
    plt.title("Shock Thickness vs. Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compression_ratio_time(t, ratio_i, ratio_e=None):
   
    plt.figure()
    plt.plot(t, ratio_i, marker='o', label="Ion Compression Ratio")
    if ratio_e is not None:
        plt.plot(t, ratio_e, marker='s', label="Electron Compression Ratio")
    plt.xlabel("Time")
    plt.ylabel("n_down / n_up")
    plt.title("Compression Ratio vs. Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mach_number_time(t, mach_number):
    
    plt.figure()
    plt.plot(t, mach_number, marker='o', color='green', label="Mach Number")
    plt.xlabel("Time")
    plt.ylabel("Mach Number (M)")
    plt.title("Shock Mach Number vs. Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_time(t, ke, pe, te):
   
    plt.figure()
    plt.semilogy(t, ke, label="Kinetic Energy", color='blue')
    plt.semilogy(t, pe, label="Potential Energy", color='orange')
    plt.semilogy(t, te, label="Total Energy", color='green')
    plt.xlabel("Time")
    plt.ylabel("Energy (log scale)")
    plt.title("Energy Evolution vs. Time")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_phase_space_1D(t_index, x_particles, vx_particles, time_val=None):
   
    x_t = x_particles[t_index]  # shape (Np,)
    v_t = vx_particles[t_index]  # shape (Np,3) or (Np,)

    if v_t.ndim > 1:
        # if we have shape (Np,3), select vx
        v_t = v_t[:, 0]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_t, v_t, s=1, alpha=0.5)
    ttl = "Phase Space: x vs. vx"
    if time_val is not None:
        ttl += f" at t={time_val:.2f}"
    plt.title(ttl)
    plt.xlabel("x")
    plt.ylabel("vx")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_vx_vy_distribution(t_index, v_particles, species="electrons", nbins=50):
   
    v_t = v_particles[t_index]  # shape (Np, 3)
    vx = v_t[:, 0]
    vy = v_t[:, 1]

    plt.figure(figsize=(6,5))
    plt.hist2d(vx, vy, bins=nbins, norm=LogNorm(), cmap="viridis")
    plt.colorbar(label="Counts (log scale)")
    plt.xlabel("vx")
    plt.ylabel("vy")
    plt.title(f"{species.capitalize()} vx-vy distribution at t_index={t_index}")
    plt.tight_layout()
    plt.show()


def plot_velocity_histogram(t_index, v_particles, component=0, species="electrons", nbins=50):
   
    v_t = v_particles[t_index]  # (Np,3) or (Np,)
    if v_t.ndim > 1:
        vel = v_t[:, component]
    else:
        vel = v_t  # if purely 1D velocity

    plt.figure()
    plt.hist(vel, bins=nbins, alpha=0.7, color='blue')
    comp_str = ["vx", "vy", "vz"][component] if v_t.ndim > 1 else "v"
    plt.xlabel(f"{comp_str}")
    plt.ylabel("Counts")
    plt.title(f"{species.capitalize()} {comp_str} distribution at t_index={t_index}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_space_time_density(t, x, n, species="ion"):
    
    Nt = len(t)
    Nx = n.shape[1]

   
    T, X = np.meshgrid(t, np.linspace(0, 1, Nx), indexing='ij')  
   
    fig, ax = plt.subplots(figsize=(7,5))
  
    cax = ax.imshow(n, aspect='auto', origin='lower', 
                    extent=[0, Nx, t[0], t[-1]], cmap='turbo')
    ax.set_xlabel("Spatial Index (not real x)")
    ax.set_ylabel("Time")
    ax.set_title(f"{species.capitalize()} Density in (x,t) space")
    fig.colorbar(cax, label="Density")
    plt.tight_layout()
    plt.show()


def plot_space_time_field(t, x, field, field_name="Ex"):
    
    fig, ax = plt.subplots(figsize=(7,5))
    cax = ax.imshow(field, aspect='auto', origin='lower',
                    extent=[0, field.shape[1], t[0], t[-1]], cmap='RdBu')
    ax.set_xlabel("Spatial Index")
    ax.set_ylabel("Time")
    ax.set_title(f"{field_name} in (x,t) space")
    fig.colorbar(cax, label=f"{field_name}")
    plt.tight_layout()
    plt.show()




def compare_shock_thickness(
    t_1d, thickness_i_1d, thickness_e_1d,
    t_1d3v, thickness_i_1d3v, thickness_e_1d3v
):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharex=False)

    # 1) Ion shock thickness subplot
    axes[0].plot(t_1d, thickness_i_1d, 'o-',  label="Ion (1D)",   color='blue')
    axes[0].plot(t_1d3v, thickness_i_1d3v, 's--', label="Ion (1D3V)", color='red')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Ion Shock Thickness")
    axes[0].set_title("Ion Shock Thickness\n(1D vs. 1D3V)")
    axes[0].grid(True)
    axes[0].legend()

    # 2) Electron shock thickness subplot
    if thickness_e_1d is not None and thickness_e_1d3v is not None:
        axes[1].plot(t_1d, thickness_e_1d, 'o-',  label="Electron (1D)",   color='green')
        axes[1].plot(t_1d3v, thickness_e_1d3v, 's--', label="Electron (1D3V)", color='orange')
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Electron Shock Thickness")
        axes[1].set_title("Electron Shock Thickness\n(1D vs. 1D3V)")
        axes[1].grid(True)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No electron thickness data", ha='center', va='center')
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


def compare_compression_ratios(
    t_1d, ratio_i_1d, ratio_e_1d,
    t_1d3v, ratio_i_1d3v, ratio_e_1d3v
):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharey=False)

    # Ion subplot
    axes[0].plot(t_1d, ratio_i_1d, 'o-',  label="Ion(1D)", color='blue')
    axes[0].plot(t_1d3v, ratio_i_1d3v, 's--', label="Ion(1D3V)", color='red')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Ion n_down / n_up")
    axes[0].set_title("Ion Compression Ratio")
    axes[0].legend()
    axes[0].grid(True)

    # Electron subplot
    axes[1].plot(t_1d, ratio_e_1d, 'o-',  label="Electron(1D)", color='green')
    axes[1].plot(t_1d3v, ratio_e_1d3v, 's--', label="Electron(1D3V)", color='orange')
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Electron n_down / n_up")
    axes[1].set_title("Electron Compression Ratio")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Comparison of Ion & Electron Compression (1D vs. 1D3V)", fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_mach_number(
    t_1d, mach_1d,
    t_1d3v, mach_1d3v
):
  
    plt.figure(figsize=(7,5))
    plt.plot(t_1d, mach_1d, 'o-',  label="Mach(1D)",   color='blue')
    plt.plot(t_1d3v, mach_1d3v, 's--', label="Mach(1D3V)", color='red')
    plt.xlabel("Time")
    plt.ylabel("Mach Number")
    plt.title("Comparison: Mach Number (1D vs. 1D3V)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_ion_energy(
    t_1d, ion_ke_1d,
    t_1d3v, ion_ke_1d3v
):
    
    plt.figure(figsize=(7,5))
    plt.plot(t_1d, ion_ke_1d, 'o-',  label="Ion KE (1D)", color='blue')
    plt.plot(t_1d3v, ion_ke_1d3v, 's--', label="Ion KE (1D3V)", color='red')
    plt.xlabel("Time")
    plt.ylabel("Ion Kinetic Energy")
    plt.title("Ion Heating: 1D vs. 1D3V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_phase_space_vx(
    t_index_1d, x_1d, v_1d,
    t_index_1d3v, x_1d3v, v_1d3v,
    species_label="Ions or Electrons",
    t_val_1d=None, t_val_1d3v=None
):
   
    fig, axes = plt.subplots(ncols=2, figsize=(12,5), sharey=True)

    # 1D data
    x_t_1d = x_1d[t_index_1d]  # shape (Np,)
    v_t_1d = v_1d[t_index_1d]
    if v_t_1d.ndim > 1:
    
        v_t_1d = v_t_1d[:, 0]

    axes[0].scatter(x_t_1d, v_t_1d, s=1, alpha=0.5, color='blue')
    title_1 = f"1D {species_label}"
    if t_val_1d is not None:
        title_1 += f" @ t={t_val_1d:.2f}"
    axes[0].set_title(title_1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("vx")
    axes[0].grid(True)


    x_t_3v = x_1d3v[t_index_1d3v]
    v_t_3v = v_1d3v[t_index_1d3v]
    if v_t_3v.ndim > 1:
        v_t_3v = v_t_3v[:, 0]

    title_2 = f"1D3V {species_label}"
    if t_val_1d3v is not None:
        title_2 += f" @ t={t_val_1d3v:.2f}"
    axes[1].scatter(x_t_3v, v_t_3v, s=1, alpha=0.5, color='red')
    axes[1].set_title(title_2)
    axes[1].set_xlabel("x")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def compare_phase_space_vy(
    t_index_1d3v, x_1d3v, v_1d3v,
    species_label="Ions",
    t_val=None
):
    
    plt.figure(figsize=(8,5))
    x_t = x_1d3v[t_index_1d3v]     
    v_t = v_1d3v[t_index_1d3v]     
    vy  = v_t[:, 1]                
    plt.scatter(x_t, vy, s=1, alpha=0.5, color='green')
    ttl = f"1D3V {species_label}: x vs. vy"
    if t_val is not None:
        ttl += f" @ t={t_val:.2f}"
    plt.title(ttl)
    plt.xlabel("x")
    plt.ylabel("vy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_field_profile(
    x_1d, E_1d, x_1d3v, E_1d3v,
    time_label_1d=None, time_label_1d3v=None,
    field_name="Ex"
):
    
    plt.figure(figsize=(8,5))
    plt.plot(x_1d, E_1d, label=f"{field_name} (1D)", color='blue')
    plt.plot(x_1d3v, E_1d3v, label=f"{field_name} (1D3V)", color='red', linestyle='--')


    title_str = f"Comparison: {field_name} profile\n(1D vs. 1D3V)"
    if time_label_1d is not None or time_label_1d3v is not None:
        title_str += "\n"
    if time_label_1d is not None:
        title_str += f"(1D t={time_label_1d})  "
    if time_label_1d3v is not None:
        title_str += f"(1D3V t={time_label_1d3v})"
    plt.title(title_str)

    plt.xlabel("x")
    plt.ylabel(field_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
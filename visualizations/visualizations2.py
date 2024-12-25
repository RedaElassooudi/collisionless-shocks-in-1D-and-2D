
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.style.use("seaborn-v0_8-colorblind")


def field_1D(time_steps, x, F, name, t):
  
    for ts in time_steps:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x[ts], F[ts], label=name, color="blue")
        ax.set_title(f"{name} at time t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


def field_ND(time_steps, x, F, component, name, t):
    
    for ts in time_steps:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x[ts], F[ts][:, component], label=name, color="blue")
        ax.set_title(f"{name} at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


def density_profiles_1D(time_steps, x, ne, ni, t):
   
    fig, axes = plt.subplots(nrows=len(time_steps), figsize=(8, 3 * len(time_steps)), sharex=True)
    if len(time_steps) == 1:
        axes = [axes] 

    for ax, ts in zip(axes, time_steps):
        ax.plot(x[ts], ne[ts], label="Electron Density", color="blue")
        ax.plot(x[ts], ni[ts], label="Ion Density", color="red")
        ax.set_title(f"Density Profiles at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Density Profiles Over Time", fontsize=16)
    fig.tight_layout()
    plt.show()


def velocity_profiles_1D(time_steps, ve, vi, t):
   
    for ts in time_steps:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ve[ts], bins=50, alpha=0.5, label="Electron Velocity", color="blue")
        ax.hist(vi[ts], bins=50, alpha=0.5, label="Ion Velocity", color="red")
        ax.set_title(f"Velocity Histogram at t = {t[ts]:.2f}")
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


def velocity_profiles_ND(time_steps, ve, vi, t, component=0):
   
    for ts in time_steps:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ve[ts][:, component], bins=50, alpha=0.5, label="Electron Velocity", color="blue")
        ax.hist(vi[ts][:, component], bins=50, alpha=0.5, label="Ion Velocity", color="red")
        ax.set_title(f"Velocity Histogram at t = {t[ts]:.2f}")
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


def phase_space_1D(time_steps, xe, ve, xi, vi, t):
    
    for ts in time_steps:
        # Electrons
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xe[ts], ve[ts], s=2, alpha=0.5, c="blue")
        ax.set_title(f"Electron Phase Space at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Velocity (v)")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

        # Ions
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xi[ts], vi[ts], s=2, alpha=0.5, c="red")
        ax.set_title(f"Ion Phase Space at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Velocity (v)")
        ax.grid(True)
        fig.tight_layout()
        plt.show()


def phase_space_ND(time_steps, xe, ve, xi, vi, t, component=0):
    
    for ts in time_steps:
        # Electrons
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xe[ts][:, 0], ve[ts][:, component], s=2, alpha=0.5, c="blue")
        ax.set_title(f"Electron Phase Space at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Velocity (v)")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

        # Ions
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xi[ts][:, 0], vi[ts][:, component], s=2, alpha=0.5, c="red")
        ax.set_title(f"Ion Phase Space at t = {t[ts]:.2f}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Velocity (v)")
        ax.grid(True)
        fig.tight_layout()
        plt.show()


def energy_evolution(t, KE, PE, TE):
   
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t, KE, label="Kinetic Energy", color="blue")
    ax.semilogy(t, PE, label="Potential Energy", color="orange")
    ax.semilogy(t, TE, label="Total Energy", color="green")
    ax.set_title("Energy Evolution Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plt.show()




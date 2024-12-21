import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_phase_space(x_e, v_e, x_max):
    # Will take a while to draw
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter([], [], s=1, alpha=0.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(min([np.min(k) for k in v_e]), max([np.max(k) for k in v_e]))
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Velocity (v)")
    ax.grid(True)

    def update(frame):
        scatter.set_offsets(np.c_[x_e[frame], v_e[frame]])
        ax.set_title(f"Electron Phase Space at Time Step {frame * 10}")
        return (scatter,)

    ani = FuncAnimation(fig, update, frames=len(x_e), interval=100)
    ani.save("phase_space_animation.gif", writer="ffmpeg", fps=30)


def field_1D(time_steps, x, F, name, t):
    for ts in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[ts], F[ts], label=name)
        plt.title(f"{name} at time {t[ts]}")
        plt.xlabel("Grid Cell")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.show()


def field_ND(time_steps, x, F, component, name, t):
    for ts in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[ts], F[ts][:, component], label=name)
        plt.title(f"{name} at time {t[ts]}")
        plt.xlabel("Grid Cell")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.show()


def density_profiles_1D(time_steps, x, ne, ni, t):
    for ts in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[ts], ne[ts], label="Electron Density")
        plt.plot(x[ts], ni[ts], label="Ion Density")
        plt.title(f"Density Profiles at time {t[ts]}")
        plt.xlabel("Grid Cell")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()


def velocity_profiles_1D(time_steps, ve, vi, t):
    for ts in time_steps:
        plt.figure(figsize=(10, 6))
        plt.hist(ve[ts], bins=50, alpha=0.5, label="Electron Velocity", color="blue")
        plt.hist(vi[ts], bins=50, alpha=0.5, label="Ion Velocity", color="red")
        plt.title(f"Velocity Histogram at time {t[ts]}")
        plt.xlabel("Velocity")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.show()


def velocity_profiles_ND(time_steps, ve, vi, t, component):
    for ts in time_steps:
        plt.figure(figsize=(10, 6))
        plt.hist(ve[ts][:, component], bins=50, alpha=0.5, label="Electron Velocity", color="blue")
        plt.hist(vi[ts][:, component], bins=50, alpha=0.5, label="Ion Velocity", color="red")
        plt.title(f"Velocity Histogram at time {t[ts]}")
        plt.xlabel("Velocity")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.show()


def phase_space_ND(time_steps, xe, ve, xi, vi, t):
    for ts in time_steps:
        # Electrons
        plt.figure(figsize=(10, 6))
        plt.scatter(xe[ts][:, 0], ve[ts][:, 0], s=1, alpha=0.5)
        plt.title(f"Electron Phase Space at time {t[ts]}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()

        # Ions
        plt.figure(figsize=(10, 6))
        plt.scatter(xi[ts][:, 0], vi[ts][:, 0], s=1, alpha=0.5, color="red")
        plt.title(f"Ion Phase Space at time {t[ts]}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()


def phase_space_1D(time_steps, xe, ve, xi, vi, t):
    for ts in time_steps:
        # Electrons
        plt.figure(figsize=(10, 6))
        plt.scatter(xe[ts], ve[ts], s=1, alpha=0.5)
        plt.title(f"Electron Phase Space at time {t[ts]}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()

        # Ions
        plt.figure(figsize=(10, 6))
        plt.scatter(xi[ts], vi[ts], s=1, alpha=0.5, color="red")
        plt.title(f"Ion Phase Space at time {t[ts]}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()


def energy_evolution(t, KE, PE, TE):
    plt.figure(figsize=(10, 6))
    plt.semilogy(t, KE, label="Kinetic Energy")
    plt.semilogy(t, PE, label="Potential Energy")
    plt.semilogy(t, TE, label="Total Energy")
    plt.title("Energy Evolution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

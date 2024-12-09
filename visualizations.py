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


def field_1D(time_steps, x, F, name):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[t], F[t], label=f"Time Step {t * 10}")
        plt.title(f"{name} at Time Step {t * 10}")
        plt.xlabel("Grid Cell")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.show()


def field_ND(time_steps, x, F, component, name):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[t], F[t][:, component], label=f"Time Step {t * 10}")
        plt.title(f"{name} at Time Step {t * 10}")
        plt.xlabel("Grid Cell")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.show()


def density_profiles_1D(time_steps, x, ne, ni):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[t], ne[t], label="Electron Density")
        plt.plot(x[t], ni[t], label="Ion Density")
        plt.title(f"Density Profiles at Time Step {t * 10}")
        plt.xlabel("Grid Cell")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()


def velocity_profiles_1D(time_steps, ve, vi):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.hist(ve[t], bins=50, alpha=0.5, label="Electron Velocity", color="blue")
        plt.hist(vi[t], bins=50, alpha=0.5, label="Ion Velocity", color="red")
        plt.title(f"Velocity Histogram at Time Step {t * 10}")
        plt.xlabel("Velocity")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.show()


def phase_space_1D(time_steps, xe, ve, xi, vi):
    for t in time_steps:
        # Electrons
        plt.figure(figsize=(10, 6))
        plt.scatter(xe[t], ve[t], s=1, alpha=0.5)
        plt.title(f"Electron Phase Space at Time Step {t * 10}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()

        # Ions
        plt.figure(figsize=(10, 6))
        plt.scatter(xi[t], vi[t], s=1, alpha=0.5, color="red")
        plt.title(f"Ion Phase Space at Time Step {t * 10}")
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity (v)")
        plt.grid(True)
        plt.show()


def energy_evolution(t, KE, PE, TE):
    plt.figure(figsize=(10, 6))
    plt.plot(t, KE, label="Kinetic Energy")
    plt.plot(t, PE, label="Potential Energy")
    plt.plot(t, TE, label="Total Energy")
    plt.title("Energy Evolution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

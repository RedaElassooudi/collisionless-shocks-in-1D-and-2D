import matplotlib.pyplot as plt


def electric_field_1D(time_steps, x, E):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(x[t], E[t], label=f"Time Step {t * 10}")
        plt.title(f"Electric Field at Time Step {t * 10}")
        plt.xlabel("Grid Cell")
        plt.ylabel("Electric Field (E)")
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


def energy_evolution(KE, PE, TE):
    plt.figure(figsize=(10, 6))
    plt.plot(KE, label="Kinetic Energy")
    plt.plot(PE, label="Potential Energy")
    plt.plot(TE, label="Total Energy")
    plt.title("Energy Evolution Over Time")
    plt.xlabel("Time Step (x10)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

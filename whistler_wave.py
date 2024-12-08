import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light in m/s
omega_p = 1e4  # Electron plasma frequency in rad/s 
Omega_e = 1e3  # Electron cyclotron frequency in rad/s

# Using np.roots to solve DR
def solve_dispersion(k):
    # Coefficients 
    A = 1
    B = omega_p**2 / np.abs(Omega_e)
    C = k**2 * c**2
    
    omega_roots = np.roots([A, -B, -C])  # Solve A*omega^2 + B*omega + C = 0
    return np.max(np.real(omega_roots))

k_values = np.linspace(1e6, 1e9, 500)  # Adjust the range of k as needed

# Calculating k
omega_values = np.array([solve_dispersion(k) for k in k_values])

# Plotting the dispersion relation
plt.figure(figsize=(8, 6))
plt.plot(k_values, omega_values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Wave Number (k) [rad/m]")
plt.ylabel("Frequency (ω) [rad/s]")
plt.title("Whistler Wave Dispersion Relation")
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:46:16 2024

@author: u0160596
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# Simulation parameters
num_particles = 20000       # Total number of particles (ions + electrons)
num_cells = 200             # Number of spatial grid cells
num_steps = 1000            # Number of time steps
qm_e = -1.0                 # Charge-to-mass ratio for electrons
qm_i = 1.0                  # Charge-to-mass ratio for ions
v_te = 1.0                  # Thermal velocity for electrons
v_ti = 0.1                  # Thermal velocity for ions
dx = 1.0                    # Spatial step size
x_max = num_cells * dx      # Maximum position value

# Shock generation parameters
bulk_velocity_e = 2.0       # Bulk velocity for electrons (towards left)
bulk_velocity_i = 0.2       # Bulk velocity for ions (towards left)

# Initialize lists for diagnostics
E_history = []
density_e_history = []
density_i_history = []
x_e_history = []
v_e_history = []
x_i_history = []
v_i_history = []
kinetic_energy_history = []
potential_energy_history = []
total_energy_history = []

# Function thqt determines the optimql timestep based on the CFL condition and the plasma frequency condition
def calculate_max_timestep(dx, v_te, qm_e):
    # CFL condition (particle shouldn't cross more than one cell per timestep)
    dt_cfl = dx / (5.0 * v_te)  # Factor 5 for safety
    
    # Plasma frequency condition
    wp = np.sqrt(abs(qm_e))  # Plasma frequency (normalized units)
    dt_wp = 0.2 / wp  # Factor 0.2 for stability
    
    # Return the more restrictive timestep
    return min(dt_cfl, dt_wp)

#Function to properly initialize velocities for the leapfrog scheme at t-dt/2.
def initialize_velocities_half_step(x_e, v_e, x_i, v_i, qm_e, qm_i, dt, dx, num_cells):
    """
    
    """
    # Calculate initial electric field
    rho = np.zeros(num_cells)
    
    # Charge assignment
    idx_e = (x_e / dx).astype(int) % num_cells
    idx_i = (x_i / dx).astype(int) % num_cells
    np.add.at(rho, idx_e, qm_e)
    np.add.at(rho, idx_i, qm_i)
    
    # Solve for initial electric field
    phi = np.zeros(num_cells)
    rho_mean = np.mean(rho)
    rho_tilde = rho - rho_mean
    
    phi[1:] = np.cumsum(rho_tilde[:-1]) * dx ** 2
    
    # Electric field calculation
    E = np.zeros(num_cells)
    E[:-1] = -(phi[1:] - phi[:-1]) / dx
    E[-1] = -(phi[0] - phi[-1]) / dx
    
    # Get field at particle positions
    E_e = E[idx_e]
    E_i = E[idx_i]
    
    # Push velocities back half timestep
    v_e_half = v_e - 0.5 * qm_e * E_e * dt
    v_i_half = v_i - 0.5 * qm_i * E_i * dt
    
    return v_e_half, v_i_half

dt = calculate_max_timestep(dx, v_te, qm_e) # Time step
print(f"Using timestep: {dt}")     

# Initialize particle positions and velocities with a bulk flow
def initialize_particles():
    # Electrons (uniformly distributed with bulk velocity towards left)
    num_electrons = num_particles // 2
    x_e = np.random.uniform(0, x_max, num_electrons)
    v_e = np.random.normal(bulk_velocity_e, v_te, num_electrons)

    # Ions (density gradient with bulk velocity towards left)
    num_ions = num_particles - num_electrons
    num_ions_left = int(num_ions * 0.7)   # 70% ions on the left half
    num_ions_right = num_ions - num_ions_left  # Remaining ions on the right half

    x_i_left = np.random.uniform(0, x_max / 2, num_ions_left)
    x_i_right = np.random.uniform(x_max / 2, x_max, num_ions_right)
    x_i = np.concatenate((x_i_left, x_i_right))

    v_i = np.random.normal(bulk_velocity_i, v_ti, num_ions)

    return x_e, v_e, x_i, v_i

# Main simulation function
def run_simulation():
    # Initialize particles
    x_e, v_e, x_i, v_i = initialize_particles()
    
    # Initialize velocities at half timestep (v^(-1/2))
    v_e, v_i = initialize_velocities_half_step(x_e, v_e, x_i, v_i, qm_e, qm_i, dt, dx, num_cells)
    
    # Spatial grid
    x_grid = np.linspace(0, x_max, num_cells + 1)[:-1]
    rho = np.zeros(num_cells)
    E = np.zeros(num_cells)
    
    for step in range(num_steps):
        # Position update (x^(n+1) = x^n + v^(n+1/2) * dt)
        x_e += v_e * dt
        x_i += v_i * dt
        
        # Apply boundary conditions
        # Reflect particles that move beyond x=0
        mask_e_left = x_e < 0
        v_e[mask_e_left] = -v_e[mask_e_left]
        x_e[mask_e_left] = -x_e[mask_e_left]
        
        mask_i_left = x_i < 0
        v_i[mask_i_left] = -v_i[mask_i_left]
        x_i[mask_i_left] = -x_i[mask_i_left]
        
        # Apply periodic boundary conditions at x_max
        x_e = x_e % x_max
        x_i = x_i % x_max
        
        # Charge assignment (NGP scheme)
        rho.fill(0)
        idx_e = (x_e / dx).astype(int) % num_cells
        idx_i = (x_i / dx).astype(int) % num_cells
        np.add.at(rho, idx_e, qm_e)
        np.add.at(rho, idx_i, qm_i)
        
        # Solve Poisson's equation
        phi = np.zeros(num_cells)
        rho_mean = np.mean(rho)
        rho_tilde = rho - rho_mean
        
        phi[1:] = np.cumsum(rho_tilde[:-1]) * dx ** 2
        
        # Electric field calculation
        E[:-1] = -(phi[1:] - phi[:-1]) / dx
        E[-1] = -(phi[0] - phi[-1]) / dx
        
        # Gather electric field at particle positions
        E_e = E[idx_e]
        E_i = E[idx_i]
        
        # Velocity update (v^(n+3/2) = v^(n+1/2) + q/m * E^(n+1) * dt)
        v_e += qm_e * E_e * dt
        v_i += qm_i * E_i * dt
        
        # Compute diagnostics
        density_e = np.zeros(num_cells)
        density_i = np.zeros(num_cells)
        np.add.at(density_e, idx_e, 1)
        np.add.at(density_i, idx_i, 1)
        
        # Compute energies (use v^(n+1/2) for kinetic energy)
        kinetic_energy_e = 0.5 * np.sum(v_e ** 2)
        kinetic_energy_i = 0.5 * np.sum(v_i ** 2)
        total_kinetic_energy = kinetic_energy_e + kinetic_energy_i
        potential_energy = 0.5 * np.sum(E ** 2) * dx
        total_energy = total_kinetic_energy + potential_energy
        
        # Store diagnostics every 10 steps
        if step % 10 == 0:
            E_history.append(E.copy())
            density_e_history.append(density_e.copy())
            density_i_history.append(density_i.copy())
            x_e_history.append(x_e.copy())
            v_e_history.append(v_e.copy())
            x_i_history.append(x_i.copy())
            v_i_history.append(v_i.copy())
            kinetic_energy_history.append(total_kinetic_energy)
            potential_energy_history.append(potential_energy)
            total_energy_history.append(total_energy)
        
        # Optional: Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")
            print(f"Total Energy: {total_energy:.6f}")
    
    return x_e, v_e, x_i, v_i, E_history

# Run the simulation
x_e, v_e, x_i, v_i, E_history = run_simulation()

# Plotting functions
def plot_electric_field(time_steps):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(E_history[t], label=f'Time Step {t * 10}')
        plt.title(f'Electric Field at Time Step {t * 10}')
        plt.xlabel('Grid Cell')
        plt.ylabel('Electric Field (E)')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_density_profiles(time_steps):
    for t in time_steps:
        plt.figure(figsize=(10, 6))
        plt.plot(density_i_history[t], label='Ion Density')
        plt.plot(density_e_history[t], label='Electron Density')
        plt.title(f'Density Profiles at Time Step {t * 10}')
        plt.xlabel('Grid Cell')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_phase_space(time_steps):
    for t in time_steps:
        # Electrons
        plt.figure(figsize=(10, 6))
        plt.scatter(x_e_history[t], v_e_history[t], s=1, alpha=0.5)
        plt.title(f'Electron Phase Space at Time Step {t * 10}')
        plt.xlabel('Position (x)')
        plt.ylabel('Velocity (v)')
        plt.grid(True)
        plt.show()

        # Ions
        plt.figure(figsize=(10, 6))
        plt.scatter(x_i_history[t], v_i_history[t], s=1, alpha=0.5, color='red')
        plt.title(f'Ion Phase Space at Time Step {t * 10}')
        plt.xlabel('Position (x)')
        plt.ylabel('Velocity (v)')
        plt.grid(True)
        plt.show()

def plot_energy_evolution():
    plt.figure(figsize=(10, 6))
    plt.plot(kinetic_energy_history, label='Kinetic Energy')
    plt.plot(potential_energy_history, label='Potential Energy')
    plt.plot(total_energy_history, label='Total Energy')
    plt.title('Energy Evolution Over Time')
    plt.xlabel('Time Step (x10)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define time steps to plot (start, middle, end)
time_steps = [0, len(E_history) // 2, -1]  # Start, middle, end

# Plot results
plot_electric_field(time_steps)
plot_density_profiles(time_steps)
plot_phase_space(time_steps)
plot_energy_evolution()
#!/usr/bin/env python
"""
Simple example: Decaying SQG turbulence simulation.

This example demonstrates basic usage of the pygSQuiG solver
for simulating decaying Surface Quasi-Geostrophic turbulence.
"""

import numpy as np
import jax.numpy as jnp
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.utils.diagnostics import compute_energy_spectrum, compute_total_energy

# Parameters
N = 256          # Grid resolution  
L = 2 * np.pi    # Domain size
alpha = 1.0      # SQG case
nu_p = 1e-4      # Hyperviscosity coefficient
p = 8            # Hyperviscosity order

# Create grid
print("Creating grid...")
grid = make_grid(N, L)

# Initialize solver
print("Initializing solver...")
solver = gSQGSolver(grid, alpha=alpha, nu_p=nu_p, p=p)

# Create initial condition with random phases
print("Setting up initial condition...")
theta0 = jnp.sin(8 * grid.x) * jnp.cos(8 * grid.y) + \
         0.5 * jnp.sin(12 * grid.x) * jnp.cos(12 * grid.y)
         
# Initialize state
state = solver.initialize(theta0)

# Get initial diagnostics
initial_energy = compute_total_energy(state['theta_hat'], grid, alpha)
print(f"Initial energy: {initial_energy:.6f}")

# Time stepping parameters
dt = 0.001
n_steps = 1000
output_interval = 100

# Run simulation
print(f"\nRunning simulation for {n_steps} steps...")
energies = [initial_energy]
times = [0.0]

for step in range(n_steps):
    # Take a time step
    state = solver.step(state, dt)
    
    # Output diagnostics
    if (step + 1) % output_interval == 0:
        energy = compute_total_energy(state['theta_hat'], grid, alpha)
        energies.append(energy)
        times.append(state['time'])
        print(f"Step {step+1}: t={state['time']:.3f}, E={energy:.6f}")

# Final diagnostics
print("\nSimulation complete!")
print(f"Energy decay: {(1 - energies[-1]/energies[0])*100:.1f}%")

# Compute final spectrum
k_bins, E_k = compute_energy_spectrum(state['theta_hat'], grid, alpha)
print(f"Peak of energy spectrum at k ≈ {k_bins[np.argmax(E_k)]:.0f}")

# Optional: Save results
print("\nTo visualize results, use pygsquig.scripts.analyse")
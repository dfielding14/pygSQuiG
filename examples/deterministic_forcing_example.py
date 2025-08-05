#!/usr/bin/env python
"""
Example: Using deterministic forcing patterns.

This example demonstrates how to use various deterministic forcing
patterns with the pygSQuiG solver.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing.deterministic_forcing import (
    TaylorGreenForcing,
    KolmogorovForcing,
    make_oscillating_forcing
)

# Parameters
N = 256          # Grid resolution
L = 2 * np.pi    # Domain size
alpha = 1.0      # SQG case

# Create grid
print("Setting up simulation with deterministic forcing...")
grid = make_grid(N, L)

# Initialize solver
solver = gSQGSolver(grid, alpha=alpha, nu_p=1e-6, p=8)

# Create deterministic forcing
# Option 1: Kolmogorov flow
forcing = KolmogorovForcing(amplitude=0.5, k=4, direction='y')

# Option 2: Taylor-Green vortex
# forcing = TaylorGreenForcing(amplitude=0.5, k=2)

# Option 3: Time-oscillating forcing
# forcing = make_oscillating_forcing(
#     base_pattern='taylor_green',
#     frequency=0.5,
#     amplitude=0.5,
#     k=2
# )

# Since the solver doesn't pass dt to forcing directly,
# we need to create a wrapper that tracks dt
class ForcingWrapper:
    def __init__(self, forcing, dt):
        self.forcing = forcing
        self.dt = dt
        
    def __call__(self, theta_hat, **kwargs):
        # Extract key and grid from kwargs
        key = kwargs.get('key', jax.random.PRNGKey(0))
        grid = kwargs['grid']
        return self.forcing(theta_hat, key, self.dt, grid)

# Initialize state
state = solver.initialize(seed=42)

# Time stepping parameters
dt = 0.01
n_steps = 1000
output_interval = 100

# Create forcing wrapper with fixed dt
forcing_wrapped = ForcingWrapper(forcing, dt)

# PRNG key (not used by deterministic forcing but needed for interface)
rng_key = jax.random.PRNGKey(12345)

print(f"\nRunning simulation for {n_steps} steps...")

# Storage for diagnostics
times = []
energies = []

for step in range(n_steps):
    # Split random key (for interface compatibility)
    rng_key, subkey = jax.random.split(rng_key)
    
    # Time step with deterministic forcing
    state = solver.step(
        state, dt,
        forcing=forcing_wrapped,
        key=subkey,
        grid=grid
    )
    
    # Diagnostics
    if step % output_interval == 0:
        # Compute energy
        theta = jnp.fft.ifft2(state['theta_hat']).real
        energy = 0.5 * jnp.mean(theta**2)
        
        times.append(state['time'])
        energies.append(float(energy))
        
        print(f"Step {step}: t={state['time']:.2f}, Energy={energy:.4f}")

print("\nSimulation complete!")

# For a more advanced example with direct forcing integration,
# you would need to extend the solver or use a custom time-stepping loop.
# The current solver interface expects forcing(theta_hat, **kwargs),
# while deterministic forcing uses forcing(theta_hat, key, dt, grid).

# Alternative approach: Custom time stepping
print("\n" + "="*50)
print("Alternative: Custom time stepping with full control")
print("="*50)

# Reset state
state2 = solver.initialize(seed=42)
theta_hat = state2['theta_hat']
t = 0.0

# Direct time stepping
for step in range(100):
    # Get forcing directly
    F_hat = forcing(theta_hat, subkey, dt, grid)
    
    # Compute RHS manually
    rhs = solver.compute_rhs(theta_hat) + F_hat
    
    # Simple Euler step for demonstration
    theta_hat = theta_hat + dt * rhs
    t += dt
    
    if step % 20 == 0:
        energy = 0.5 * jnp.mean(jnp.abs(jnp.fft.ifft2(theta_hat))**2)
        print(f"Step {step}: t={t:.2f}, Energy={energy:.4f}")

print("\nExample complete! This demonstrates two approaches:")
print("1. Using a wrapper to adapt forcing to solver interface")
print("2. Custom time stepping with direct forcing control")
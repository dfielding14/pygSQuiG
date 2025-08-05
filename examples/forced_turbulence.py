#!/usr/bin/env python
"""
Example: Forced-dissipative SQG turbulence.

This example demonstrates how to set up a forced-dissipative
turbulence simulation with ring forcing and combined damping.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing.ring_forcing import RingForcing
from pygsquig.forcing.damping import CombinedDamping
from pygsquig.utils.diagnostics import (
    compute_energy_spectrum,
    compute_total_energy,
    compute_enstrophy,
)

# Parameters
N = 512  # Grid resolution (use power of 2)
L = 2 * np.pi  # Domain size
alpha = 1.0  # SQG case

# Forcing parameters
kf = 40.0  # Forcing wavenumber
dk = 2.0  # Forcing bandwidth
epsilon = 0.1  # Energy injection rate

# Damping parameters
mu = 0.1  # Large-scale damping coefficient
nu_p = 1e-8  # Hyperviscosity coefficient
p = 8  # Hyperviscosity order

# Create grid
print("Setting up simulation...")
grid = make_grid(N, L)

# Initialize solver
solver = gSQGSolver(grid, alpha=alpha, nu_p=nu_p, p=p)

# Create forcing and damping
forcing = RingForcing(kf=kf, dk=dk, epsilon=epsilon)
damping = CombinedDamping(mu=mu, kf=kf, nu_p=nu_p, p=p)

# Initialize with small random perturbations
state = solver.initialize(seed=42)

# PRNG key for stochastic forcing
rng_key = jax.random.PRNGKey(12345)

# Time stepping
dt = 0.001
n_steps = 10000
output_interval = 1000

print(f"\nRunning forced simulation for {n_steps} steps...")
print("This simulates statistically stationary turbulence.\n")

# Storage for diagnostics
times = []
energies = []
enstrophies = []
injection_rates = []

for step in range(n_steps):
    # Split random key for this step
    rng_key, subkey = jax.random.split(rng_key)

    # Time step with forcing and damping
    state = solver.step(state, dt, forcing=forcing, damping=damping, key=subkey, grid=grid)

    # Diagnostics every output_interval steps
    if step % output_interval == 0:
        # Compute diagnostics
        energy = compute_total_energy(state["theta_hat"], grid, alpha)
        enstrophy = compute_enstrophy(state["theta_hat"], grid, alpha)

        # Get forcing diagnostics
        forcing_hat = forcing(state["theta_hat"], subkey, dt, grid)
        forcing_diag = forcing.get_diagnostics(state["theta_hat"], forcing_hat, grid)

        # Store
        times.append(state["time"])
        energies.append(energy)
        enstrophies.append(enstrophy)
        injection_rates.append(forcing_diag["injection_rate"])

        print(f"Step {step}: t={state['time']:.2f}")
        print(f"  Energy: {energy:.4f}")
        print(f"  Enstrophy: {enstrophy:.4f}")
        print(f"  Injection rate: {forcing_diag['injection_rate']:.4f}")
        print()

# Final analysis
print("Simulation complete!")
print("\nFinal statistics:")
print(f"Mean energy: {np.mean(energies[-5:]):.4f}")
print(f"Mean injection rate: {np.mean(injection_rates[-5:]):.4f}")

# Compute spectrum
k_bins, E_k = compute_energy_spectrum(state["theta_hat"], grid, alpha)

# Find spectral slope in inertial range
k_min, k_max = kf * 1.5, N / 4  # Inertial range estimate
mask = (k_bins > k_min) & (k_bins < k_max)
if np.sum(mask) > 5:
    k_inertial = k_bins[mask]
    E_inertial = E_k[mask]

    # Log-log fit
    coeffs = np.polyfit(np.log(k_inertial), np.log(E_inertial + 1e-20), 1)
    slope = coeffs[0]
    print(f"\nSpectral slope in range k=[{k_min:.0f}, {k_max:.0f}]: {slope:.2f}")
    print(f"Theoretical SQG slope: -5/3 = {-5/3:.2f}")

print("\nTo visualize the results, save outputs and use analyse.py")

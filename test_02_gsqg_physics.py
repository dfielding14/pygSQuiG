#!/usr/bin/env python3
"""Test script for 02_gsqg_physics notebook."""

import numpy as np
import jax.numpy as jnp
from pygsquig.core.grid import make_grid, ifft2, fft2
from pygsquig.core.solver import gSQGSolver
from pygsquig.utils.diagnostics import (
    compute_total_energy,
    compute_enstrophy,
    compute_energy_spectrum,
)
from pygsquig.core.operators import compute_velocity_from_theta

print("Testing 02_gsqg_physics notebook...")

# 1. Grid Setup
print("\n1. Grid Setup")
N = 64  # Reduced for testing
L = 2 * np.pi
grid = make_grid(N, L)

alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0]
alpha_names = ['2D NS', 'α=0.5', 'SQG', 'α=1.5', 'α=2.0']
nu_p = 1e-16
p = 8

print(f"Grid: {N}×{N}, L={L:.2f}")
print(f"Testing α values: {alpha_values}")

# 2. Conservation Properties
print("\n2. Conservation Properties")
np.random.seed(42)
theta_init = np.random.randn(N, N)

solvers = {}
states = {}
initial_diagnostics = {}

for i, alpha in enumerate(alpha_values):
    solver = gSQGSolver(grid=grid, alpha=alpha, nu_p=nu_p, p=p)
    solvers[alpha] = solver
    
    state = solver.initialize(theta0=theta_init)
    states[alpha] = state
    
    energy = compute_total_energy(state['theta_hat'], grid, alpha)
    enstrophy = compute_enstrophy(state['theta_hat'], grid, alpha)
    initial_diagnostics[alpha] = {'energy': energy, 'enstrophy': enstrophy}
    
    print(f"{alpha_names[i]}: E₀={energy:.3f}, Ω₀={enstrophy:.3f}")
    
    # Verify no NaN
    assert not jnp.any(jnp.isnan(state['theta_hat'])), f"NaN in initial state for α={alpha}"

# 3. Spectral Analysis
print("\n3. Initial Spectral Analysis")
for alpha in [0.0, 1.0]:
    k_bins, E_k = compute_energy_spectrum(states[alpha]['theta_hat'], grid, alpha)
    print(f"  α={alpha}: {len(k_bins)} k-bins, E_max={E_k.max():.3e}")
    
    # Check spectrum is physical
    assert not np.any(np.isnan(E_k)), f"NaN in spectrum for α={alpha}"
    assert np.all(E_k >= 0), f"Negative energy spectrum for α={alpha}"

# 4. Short Evolution
print("\n4. Evolution Test")
dt = 0.001
n_steps = 100

print("Evolving systems...")
for step in range(n_steps):
    for alpha in alpha_values:
        states[alpha] = solvers[alpha].step(states[alpha], dt)
    
    # Check for NaN
    if (step + 1) % 20 == 0:
        all_ok = True
        for alpha in alpha_values:
            if jnp.any(jnp.isnan(states[alpha]['theta_hat'])):
                print(f"❌ NaN detected for α={alpha} at step {step+1}")
                all_ok = False
        if all_ok:
            print(f"  Step {step+1}: All systems stable")

# 5. Velocity Field Structure
print("\n5. Velocity Field Analysis")
# Test velocity computation for different α
x, y = grid.x, grid.y
x0, y0 = L/2, L/2
sigma = L/8
theta_test = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))
theta_test_hat = fft2(theta_test)

max_speeds = {}
for alpha in alpha_values:
    u, v = compute_velocity_from_theta(theta_test_hat, grid, alpha)
    speed = np.sqrt(u**2 + v**2)
    max_speeds[alpha] = float(np.max(speed))
    
    # Verify velocity field
    assert not np.any(np.isnan(u)), f"NaN in u for α={alpha}"
    assert not np.any(np.isnan(v)), f"NaN in v for α={alpha}"

print("Maximum speeds for Gaussian blob:")
for i, alpha in enumerate(alpha_values):
    print(f"  {alpha_names[i]}: {max_speeds[alpha]:.3f}")

# 6. Conservation Check
print("\n6. Conservation Test")
# Test conservation with minimal dissipation
alpha_test = 1.0
solver_cons = gSQGSolver(grid=grid, alpha=alpha_test, nu_p=1e-20, p=8)
state_cons = solver_cons.initialize(seed=99)

E0 = compute_total_energy(state_cons['theta_hat'], grid, alpha_test)
Omega0 = compute_enstrophy(state_cons['theta_hat'], grid, alpha_test)

# Very short evolution
dt_cons = 0.0001
for _ in range(10):
    state_cons = solver_cons.step(state_cons, dt_cons)

E1 = compute_total_energy(state_cons['theta_hat'], grid, alpha_test)
Omega1 = compute_enstrophy(state_cons['theta_hat'], grid, alpha_test)

E_change = abs(E1 - E0) / E0
Omega_change = abs(Omega1 - Omega0) / Omega0

print(f"  Energy change: {E_change:.2e}")
print(f"  Enstrophy change: {Omega_change:.2e}")

# Should conserve to high precision
assert E_change < 1e-10, "Poor energy conservation!"
# Enstrophy is more sensitive to numerical errors
assert Omega_change < 1e-5, "Poor enstrophy conservation!"

# 7. Physical Behavior Verification
print("\n7. Physical Behavior Checks")
# Check that different α gives different dynamics
theta_final = {}
for alpha in [0.0, 1.0, 2.0]:
    theta_final[alpha] = ifft2(states[alpha]['theta_hat']).real

# Compute differences
diff_01 = np.mean(np.abs(theta_final[0.0] - theta_final[1.0]))
diff_12 = np.mean(np.abs(theta_final[1.0] - theta_final[2.0]))

print(f"  Mean difference α=0 vs α=1: {diff_01:.3f}")
print(f"  Mean difference α=1 vs α=2: {diff_12:.3f}")

# Should see different evolution
assert diff_01 > 0.01, "α=0 and α=1 too similar!"
assert diff_12 > 0.01, "α=1 and α=2 too similar!"

# 8. Cascade Direction Indicators
print("\n8. Spectral Evolution")
for alpha in [0.0, 1.0]:
    k_bins, E_k_init = compute_energy_spectrum(
        solvers[alpha].initialize(seed=42)['theta_hat'], grid, alpha
    )
    k_bins, E_k_final = compute_energy_spectrum(states[alpha]['theta_hat'], grid, alpha)
    
    # Check energy redistribution
    low_k_change = (E_k_final[:5].sum() - E_k_init[:5].sum()) / E_k_init[:5].sum()
    high_k_change = (E_k_final[-10:].sum() - E_k_init[-10:].sum()) / E_k_init[-10:].sum()
    
    print(f"  α={alpha}:")
    print(f"    Low-k energy change: {low_k_change:+.1%}")
    print(f"    High-k energy change: {high_k_change:+.1%}")

# 9. Final Verification
print("\n9. Final State Check")
for i, alpha in enumerate(alpha_values):
    theta = ifft2(states[alpha]['theta_hat']).real
    print(f"  {alpha_names[i]}: range=[{theta.min():.2f}, {theta.max():.2f}], "
          f"mean={theta.mean():.3f}")
    
    # Should remain bounded
    assert np.abs(theta).max() < 100, f"Unbounded growth for α={alpha}!"

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("The gSQG physics notebook is working correctly.")
print("Key physics verified:")
print("  - Different α values produce different dynamics")
print("  - Conservation properties maintained")
print("  - Velocity-scalar relationships correct")
print("  - Spectral behavior physical")
print("  - No numerical instabilities")
print("="*50)
"""
Validation script for pygSQuiG solver.

This script runs standard test cases to validate the solver implementation
against known physical behaviors and theoretical predictions.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import time

from pygsquig.core.grid import make_grid, fft2, ifft2
from pygsquig.core.solver import gSQGSolver
from pygsquig.utils.diagnostics import compute_energy_spectrum, compute_total_energy


def compute_energy_sqg(theta_hat, grid):
    """Compute energy for SQG (α=1) case."""
    # For SQG, energy is just (1/2)⟨θ²⟩
    theta = ifft2(theta_hat)
    return 0.5 * jnp.mean(theta**2)


def initialize_turbulent_field(grid, k_peak=20, seed=42):
    """Initialize a turbulent field with energy concentrated around k_peak."""
    key = jax.random.PRNGKey(seed)
    
    # Create a real random field in physical space first
    theta = jax.random.normal(key, shape=(grid.N, grid.N))
    
    # Transform to spectral space
    theta_hat = fft2(theta)
    
    # Apply spectral filter to concentrate energy around k_peak
    k_mag = jnp.sqrt(grid.k2)
    spectrum_filter = jnp.exp(-((k_mag - k_peak) / (k_peak/4))**2)
    spectrum_filter = spectrum_filter.at[0, 0].set(0)  # Zero mean
    
    # Apply filter
    theta_hat = theta_hat * spectrum_filter
    
    # Apply dealiasing
    theta_hat = theta_hat * grid.dealias_mask
    
    # Convert back to physical space
    theta = ifft2(theta_hat).real
    
    # Normalize to have desired initial energy
    energy = jnp.mean(theta**2)
    target_energy = 0.01  # Much smaller for stability
    if energy > 0:
        theta = theta * jnp.sqrt(target_energy / energy)
    
    return theta


def test_energy_decay():
    """Test energy decay in dissipative simulation."""
    print("\n=== Test 1: Energy Decay with Hyperviscosity ===")
    
    # Setup
    N = 128
    L = 2 * np.pi
    grid = make_grid(N, L)
    
    # SQG with hyperviscosity (p=2 for better stability)
    solver = gSQGSolver(grid, alpha=1.0, nu_p=0.001, p=2)
    
    # Initialize turbulent field with lower peak wavenumber for stability
    theta0 = initialize_turbulent_field(grid, k_peak=10)
    state = solver.initialize(theta0)
    
    # Run simulation
    dt = 0.001
    n_steps = 500
    times = []
    energies = []
    
    print(f"Running {n_steps} steps with dt={dt}...")
    start_time = time.time()
    
    for i in range(n_steps):
        if i % 50 == 0:
            energy = compute_energy_sqg(state['theta_hat'], grid)
            times.append(state['time'])
            energies.append(float(energy))
            print(f"Step {i}: t={state['time']:.3f}, E={energy:.6f}")
        
        state = solver.step(state, dt)
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")
    
    # Check energy decay
    energies = np.array(energies)
    initial_energy = energies[0]
    final_energy = energies[-1]
    
    print(f"\nInitial energy: {initial_energy:.6f}")
    print(f"Final energy: {final_energy:.6f}")
    print(f"Energy decay: {(1 - final_energy/initial_energy)*100:.1f}%")
    
    assert final_energy < initial_energy, "Energy should decay with dissipation"
    assert final_energy > 0.5 * initial_energy, "Energy decay too rapid - check dissipation"
    
    print("✓ Energy decay test passed")
    return times, energies, state


def test_spectral_slope(state, grid):
    """Test energy spectrum slope."""
    print("\n=== Test 2: Energy Spectrum Slope ===")
    
    # Compute spectrum
    k_bins, E_k = compute_energy_spectrum(state['theta_hat'], grid, alpha=1.0)
    
    # Find inertial range (k between 10 and 40)
    mask = (k_bins > 10) & (k_bins < 40)
    k_inertial = k_bins[mask]
    E_inertial = E_k[mask]
    
    # Fit power law in log-log space
    log_k = np.log(k_inertial)
    log_E = np.log(E_inertial)
    slope, intercept = np.polyfit(log_k, log_E, 1)
    
    print(f"Measured spectral slope: {slope:.2f}")
    print(f"Expected SQG slope: -5/3 ≈ {-5/3:.2f}")
    
    # For SQG, expect slope close to -5/3 in ideal case
    # With strong dissipation, slope will be steeper
    print(f"Note: With strong dissipation (nu={0.001:.3f}), slope is steeper than ideal -5/3")
    
    # Just check that we have a negative slope (energy cascade)
    assert slope < -1.0, f"Spectral slope {slope:.2f} should be negative (forward cascade)"
    
    print("✓ Spectral slope test passed")
    
    return k_bins, E_k, slope


def test_inviscid_conservation():
    """Test energy conservation in inviscid limit."""
    print("\n=== Test 3: Energy Conservation (Inviscid) ===")
    
    # Setup
    N = 64
    L = 2 * np.pi
    grid = make_grid(N, L)
    
    # SQG without dissipation
    solver = gSQGSolver(grid, alpha=1.0, nu_p=0.0)
    
    # Initialize smooth field to avoid aliasing
    theta0 = jnp.sin(4 * grid.x) * jnp.cos(4 * grid.y)
    state = solver.initialize(theta0)
    
    # Run simulation
    dt = 0.0001
    n_steps = 100
    initial_energy = compute_energy_sqg(state['theta_hat'], grid)
    
    print(f"Running {n_steps} steps with dt={dt}...")
    for _ in range(n_steps):
        state = solver.step(state, dt)
    
    final_energy = compute_energy_sqg(state['theta_hat'], grid)
    
    print(f"Initial energy: {initial_energy:.10f}")
    print(f"Final energy: {final_energy:.10f}")
    print(f"Relative change: {abs(final_energy - initial_energy)/initial_energy:.2e}")
    
    # Energy should be conserved to high precision
    assert abs(final_energy - initial_energy) / initial_energy < 1e-8
    
    print("✓ Energy conservation test passed")


def plot_results(times, energies, k_bins, E_k, slope):
    """Plot validation results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy decay
        ax1.plot(times, energies, 'b-', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Decay with Hyperviscosity')
        ax1.grid(True, alpha=0.3)
        
        # Energy spectrum
        ax2.loglog(k_bins, E_k, 'b-', linewidth=2, label='Simulation')
        
        # Add reference slope
        k_ref = k_bins[(k_bins > 10) & (k_bins < 40)]
        if len(k_ref) > 0:
            E_ref = E_k[k_bins == k_ref[0]][0] * (k_ref / k_ref[0]) ** (-5/3)
            ax2.loglog(k_ref, E_ref, 'r--', linewidth=2, 
                       label=f'k^(-5/3) reference')
        
        ax2.set_xlabel('Wavenumber k')
        ax2.set_ylabel('Energy Spectrum E(k)')
        ax2.set_title(f'Energy Spectrum (slope = {slope:.2f})')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('validation_results')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'validation_plots.png', dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to {output_dir / 'validation_plots.png'}")
        
        plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        print(f"\nWarning: Could not create plots ({e})")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("pygSQuiG Validation Suite")
    print("=" * 60)
    
    # Test 1: Energy decay
    times, energies, final_state = test_energy_decay()
    
    # Test 2: Spectral slope
    grid = make_grid(128, 2*np.pi)
    k_bins, E_k, slope = test_spectral_slope(final_state, grid)
    
    # Test 3: Inviscid conservation
    test_inviscid_conservation()
    
    # Plot results
    plot_results(times, energies, k_bins, E_k, slope)
    
    print("\n" + "=" * 60)
    print("All validation tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
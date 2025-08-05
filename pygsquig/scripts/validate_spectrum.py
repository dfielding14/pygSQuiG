"""Validation script for testing SQG turbulence spectrum.

This script runs a proper forced-dissipative SQG simulation to verify
the k^(-5/3) spectrum develops in the inertial range.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing import RingForcing
from pygsquig.utils.diagnostics import compute_energy_spectrum, compute_total_energy


def run_forced_sqg_simulation():
    """Run forced SQG simulation to develop turbulent cascade."""
    print("=== Forced SQG Turbulence Test ===")
    print("Testing for k^(-5/3) spectrum in inertial range")

    # Setup - larger domain and resolution for better inertial range
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # SQG with weaker hyperviscosity at small scales
    # Use higher order (p=8) to minimize effect on inertial range
    solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-12, p=8)

    # Ring forcing at intermediate scales
    kf = 30.0  # Force at wavenumber 30
    dk = 3.0  # Width of forcing ring
    epsilon = 0.5  # Energy injection rate
    forcing = RingForcing(kf=kf, dk=dk, epsilon=epsilon)

    # Initialize with small random perturbation
    key = jax.random.PRNGKey(42)
    theta0 = 0.01 * jax.random.normal(key, shape=(N, N))
    state = solver.initialize(theta0)

    # Time stepping parameters
    dt = 0.0005
    n_spinup = 10000  # Spin-up steps
    n_stats = 5000  # Steps for statistics

    print(f"\nParameters:")
    print(f"  Resolution: {N}x{N}")
    print(f"  Forcing at kf={kf:.0f} with ε={epsilon}")
    print(f"  Hyperviscosity: ν_p={solver.nu_p:.2e}, p={solver.p}")
    print(f"  Time step: dt={dt}")

    # Spin-up phase
    print(f"\nSpin-up phase: {n_spinup} steps...")
    start_time = time.time()

    for i in range(n_spinup):
        key, subkey = jax.random.split(key)
        forcing_fn = lambda theta_hat: forcing(theta_hat, subkey, dt, grid)
        state = solver.step(state, dt, forcing=forcing_fn)

        if i % 1000 == 0:
            energy = compute_total_energy(state["theta_hat"], grid, solver.alpha)
            print(f"  Step {i}: E={energy:.4f}")

    spin_up_time = time.time() - start_time
    print(f"Spin-up completed in {spin_up_time:.1f} seconds")

    # Collect statistics
    print(f"\nCollecting statistics: {n_stats} steps...")
    spectra = []

    for i in range(n_stats):
        key, subkey = jax.random.split(key)
        forcing_fn = lambda theta_hat: forcing(theta_hat, subkey, dt, grid)
        state = solver.step(state, dt, forcing=forcing_fn)

        if i % 100 == 0:
            k, E_k = compute_energy_spectrum(state["theta_hat"], grid, solver.alpha)
            spectra.append(E_k)

            if i % 1000 == 0:
                energy = compute_total_energy(state["theta_hat"], grid, solver.alpha)
                print(f"  Step {n_spinup + i}: E={energy:.4f}")

    stats_time = time.time() - start_time - spin_up_time
    print(f"Statistics collection completed in {stats_time:.1f} seconds")

    # Average spectrum
    E_k_avg = np.mean(spectra, axis=0)

    # Identify inertial range
    # Between forcing scale (kf) and dissipation scale
    k_diss = N / 3  # Rough estimate of dissipation scale
    inertial_start = int(kf * 1.5)  # Start above forcing
    inertial_end = int(k_diss * 0.7)  # End before dissipation

    # Fit power law in inertial range
    mask = (k > k[inertial_start]) & (k < k[inertial_end])
    k_inertial = k[mask]
    E_inertial = E_k_avg[mask]

    # Log-log fit
    log_k = np.log(k_inertial)
    log_E = np.log(E_inertial + 1e-20)
    slope, intercept = np.polyfit(log_k, log_E, 1)

    print(f"\nResults:")
    print(f"  Inertial range: k ∈ [{k[inertial_start]:.0f}, {k[inertial_end]:.0f}]")
    print(f"  Measured slope: {slope:.3f}")
    print(f"  Expected SQG slope: -5/3 ≈ {-5/3:.3f}")
    print(f"  Relative error: {abs(slope - (-5/3))/abs(-5/3)*100:.1f}%")

    # Check if slope is reasonably close to -5/3
    tolerance = 0.2  # Allow 20% deviation
    expected = -5 / 3
    if abs(slope - expected) / abs(expected) < tolerance:
        print("✓ Spectrum test PASSED")
        success = True
    else:
        print("✗ Spectrum test FAILED - slope too far from -5/3")
        success = False

    return k, E_k_avg, slope, success


def plot_spectrum(k, E_k, slope):
    """Plot the energy spectrum with reference slopes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Main spectrum
    ax.loglog(k[1:], E_k[1:], "b-", linewidth=2, label="E(k)")

    # Reference slopes
    k_ref = k[20:60]  # Range for reference lines

    # k^(-5/3) reference
    E_ref_53 = E_k[30] * (k_ref / k_ref[0]) ** (-5 / 3)
    ax.loglog(k_ref, E_ref_53, "r--", linewidth=1.5, label=r"$k^{-5/3}$", alpha=0.7)

    # k^(-3) reference (steep)
    E_ref_3 = E_k[30] * (k_ref / k_ref[0]) ** (-3)
    ax.loglog(k_ref, E_ref_3, "g--", linewidth=1.5, label=r"$k^{-3}$", alpha=0.7)

    # Measured slope
    k_fit = k[20:80]
    E_fit = E_k[30] * (k_fit / k_fit[0]) ** slope
    ax.loglog(k_fit, E_fit, "k:", linewidth=2, label=f"Measured: $k^{{{slope:.2f}}}$")

    # Annotations
    ax.axvline(30, color="gray", linestyle=":", alpha=0.5, label="Forcing")
    ax.axvline(85, color="gray", linestyle="--", alpha=0.5, label="Dissipation")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy Spectrum E(k)")
    ax.set_title("SQG Turbulence Energy Spectrum")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left")

    ax.set_xlim(1, 120)
    ax.set_ylim(1e-6, 1e0)

    plt.tight_layout()

    # Save figure
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "sqg_spectrum_validation.png", dpi=150, bbox_inches="tight")
    print(f"\nSpectrum plot saved to {output_dir / 'sqg_spectrum_validation.png'}")
    plt.close(fig)


def main():
    """Run spectrum validation test."""
    print("=" * 60)
    print("SQG Turbulence Spectrum Validation")
    print("=" * 60)

    # Run simulation
    k, E_k, slope, success = run_forced_sqg_simulation()

    # Plot results
    plot_spectrum(k, E_k, slope)

    print("\n" + "=" * 60)
    if success:
        print("Validation PASSED! ✓")
    else:
        print("Validation FAILED! ✗")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

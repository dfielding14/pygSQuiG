"""
Example of stochastic forcing patterns in pygSQuiG.

This script demonstrates various random forcing patterns including
white noise, colored noise, vortex injection, and OU processes.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.diagnostics import compute_energy_spectrum, compute_total_energy
from pygsquig.core.adaptive_solver import AdaptivegSQGSolver
from pygsquig.core.adaptive_timestep import CFLConfig
from pygsquig.forcing.stochastic_forcing import (
    WhiteNoiseForcing,
    ColoredNoiseForcing,
    StochasticVortexForcing,
    OrnsteinUhlenbeckForcing,
    create_combined_stochastic_forcing,
)


def example_white_noise():
    """Demonstrate white noise forcing."""
    print("\n=== White Noise Forcing Example ===")

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Create solver with adaptive timestepping
    cfl_config = CFLConfig(cfl_safety=0.8, dt_max=0.01)
    solver = AdaptivegSQGSolver(grid, alpha=1.0, nu_p=1e-6, p=8, cfl_config=cfl_config)

    # White noise forcing in inertial range
    forcing = WhiteNoiseForcing(amplitude=0.5, k_min=20.0, k_max=40.0, isotropy=True)

    # Initialize
    state = solver.initialize(seed=42)

    # Evolve with forcing
    print("Evolving with white noise forcing...")

    # Setup for animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Storage for diagnostics
    times = []
    energies = []

    def forcing_wrapper(state, dt, key):
        """Wrapper to match solver interface."""
        return forcing(state["theta_hat"], key, dt, grid)

    # Animation function
    def animate(frame):
        nonlocal state

        # Generate random key for this step
        key = jax.random.PRNGKey(frame)

        # Step with forcing
        state, info = solver.step(state, forcing=forcing_wrapper, key=key)

        # Diagnostics
        times.append(state["time"])
        energies.append(compute_total_energy(state["theta_hat"], grid))

        # Plot physical space
        theta = grid.ifft(state["theta_hat"]).real
        im1 = axes[0].imshow(theta, cmap="RdBu_r", extent=[0, L, 0, L], vmin=-3, vmax=3)
        axes[0].set_title(f'θ at t={state["time"]:.2f}')
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # Plot energy spectrum
        k_bins, spectrum = compute_energy_spectrum(state["theta_hat"], grid)
        axes[1].clear()
        axes[1].loglog(k_bins, spectrum, "b-", label="Spectrum")
        axes[1].loglog(k_bins, k_bins ** (-5 / 3), "k--", alpha=0.5, label="k^{-5/3}")

        # Mark forcing range
        axes[1].axvspan(forcing.k_min, forcing.k_max, alpha=0.2, color="red", label="Forcing")

        axes[1].set_xlabel("k")
        axes[1].set_ylabel("E(k)")
        axes[1].set_title("Energy Spectrum")
        axes[1].set_xlim(1, N // 2)
        axes[1].set_ylim(1e-8, 1e0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot energy evolution
        axes[2].clear()
        axes[2].plot(times, energies, "g-")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Total Energy")
        axes[2].set_title("Energy Evolution")
        axes[2].grid(True, alpha=0.3)

        return [im1]

    # Create animation
    anim = FuncAnimation(fig, animate, frames=500, interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    return state, times, energies


def example_colored_noise():
    """Demonstrate colored noise forcing."""
    print("\n=== Colored Noise Forcing Example ===")

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    solver = gSQGSolver(grid, alpha=0.5, nu_p=1e-7, p=8)

    # Red noise forcing peaked at large scales
    forcing = ColoredNoiseForcing(
        amplitude=0.3, spectral_slope=-2.0, k_peak=5.0, k_width=3.0  # Red noise  # Large scales
    )

    # Initialize
    state = solver.initialize(seed=42)
    dt = 0.001

    # Storage
    spectra_history = []
    save_interval = 100

    print("Evolving with colored noise forcing...")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def forcing_wrapper(state, dt, key):
        return forcing(state["theta_hat"], key, dt, grid)

    # Evolve
    key = jax.random.PRNGKey(123)
    for step in range(2000):
        key, subkey = jax.random.split(key)
        state = solver.step(state, dt, forcing=forcing_wrapper, key=subkey)

        if step % save_interval == 0:
            k_bins, spectrum = compute_energy_spectrum(state["theta_hat"], grid)
            spectra_history.append(spectrum)

            # Update plots
            ax1.clear()
            theta = grid.ifft(state["theta_hat"]).real
            im = ax1.imshow(theta, cmap="RdBu_r", extent=[0, L, 0, L])
            ax1.set_title(f'θ at t={state["time"]:.2f}')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            ax2.clear()
            # Plot spectrum evolution
            for i, spec in enumerate(spectra_history[-5:]):
                alpha = 0.3 + 0.7 * i / min(4, len(spectra_history) - 1)
                ax2.loglog(k_bins, spec, alpha=alpha, color="blue")

            ax2.loglog(k_bins, k_bins ** (-3), "k--", alpha=0.5, label="k^{-3}")
            ax2.axvline(forcing.k_peak, color="red", linestyle=":", label="k_peak")
            ax2.set_xlabel("k")
            ax2.set_ylabel("E(k)")
            ax2.set_title("Energy Spectrum Evolution")
            ax2.set_xlim(1, N // 2)
            ax2.set_ylim(1e-8, 1e0)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.pause(0.01)

    plt.tight_layout()
    plt.show()

    return state, spectra_history


def example_vortex_injection():
    """Demonstrate stochastic vortex injection."""
    print("\n=== Vortex Injection Example ===")

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-5, p=4)

    # Vortex injection forcing
    forcing = StochasticVortexForcing(
        amplitude=2.0,  # Strong vortices
        vortex_size=0.15,  # 15% of domain
        injection_rate=2.0,  # 2 vortices per unit time average
        vortex_strength_std=0.3,
    )

    # Initialize with weak random field
    state = solver.initialize(seed=42, energy=0.01)
    dt = 0.01

    print("Injecting random vortices...")

    # Create figure for visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def forcing_wrapper(state, dt, key):
        return forcing(state["theta_hat"], key, dt, grid)

    # Track vortex injection events
    vortex_times = []
    vortex_counts = []

    # Animation
    key = jax.random.PRNGKey(456)
    for step in range(500):
        key, subkey = jax.random.split(key)

        # Apply forcing and step
        force_increment = forcing(state["theta_hat"], subkey, dt, grid)
        state = solver.step(state, dt, forcing=forcing_wrapper, key=subkey)

        # Detect injection events
        force_max = jnp.max(jnp.abs(grid.ifft(force_increment)))
        if force_max > 0.1:  # Threshold for detection
            vortex_times.append(state["time"])

        if step % 20 == 0:
            # Update plots
            theta = grid.ifft(state["theta_hat"]).real

            # Theta field
            axes[0, 0].clear()
            im = axes[0, 0].imshow(theta, cmap="RdBu_r", extent=[0, L, 0, L], vmin=-5, vmax=5)
            axes[0, 0].set_title(f'θ at t={state["time"]:.2f}')
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")

            # Vorticity (approximation)
            axes[0, 1].clear()
            vort = -theta  # For SQG
            axes[0, 1].imshow(vort, cmap="RdBu_r", extent=[0, L, 0, L], vmin=-5, vmax=5)
            axes[0, 1].set_title("Vorticity")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("y")

            # Energy spectrum
            axes[1, 0].clear()
            k_bins, spectrum = compute_energy_spectrum(state["theta_hat"], grid)
            axes[1, 0].loglog(k_bins, spectrum, "b-")
            axes[1, 0].loglog(k_bins, k_bins ** (-5 / 3), "k--", alpha=0.5, label="k^{-5/3}")
            axes[1, 0].set_xlabel("k")
            axes[1, 0].set_ylabel("E(k)")
            axes[1, 0].set_title("Energy Spectrum")
            axes[1, 0].set_xlim(1, N // 2)
            axes[1, 0].set_ylim(1e-6, 1e1)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Injection timeline
            axes[1, 1].clear()
            if vortex_times:
                axes[1, 1].scatter(vortex_times, range(len(vortex_times)), color="red", s=20)
            axes[1, 1].set_xlabel("Time")
            axes[1, 1].set_ylabel("Injection Event #")
            axes[1, 1].set_title("Vortex Injection Timeline")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    plt.show()

    print(f"Total vortex injections: {len(vortex_times)}")
    print(f"Average injection rate: {len(vortex_times)/state['time']:.2f} per unit time")

    return state, vortex_times


def example_ou_forcing():
    """Demonstrate Ornstein-Uhlenbeck forcing."""
    print("\n=== Ornstein-Uhlenbeck Forcing Example ===")

    # Setup
    N = 128
    L = 2 * np.pi
    grid = make_grid(N, L)

    solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-6, p=8)

    # OU forcing with temporal correlations
    forcing = OrnsteinUhlenbeckForcing(
        amplitude=0.3, correlation_time=0.5, k_min=10.0, k_max=30.0  # Correlation time
    )

    # Initialize
    state = solver.initialize(seed=42)
    dt = 0.01

    print("Evolving with temporally correlated forcing...")

    # Track forcing correlation
    forcing_history = []
    correlation_function = []

    def forcing_wrapper(state, dt, key):
        force = forcing(state["theta_hat"], key, dt, grid)
        forcing_history.append(force.copy())
        return force

    # Evolve and compute correlations
    key = jax.random.PRNGKey(789)
    n_steps = 1000

    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        state = solver.step(state, dt, forcing=forcing_wrapper, key=subkey)

        # Compute correlation with initial forcing
        if len(forcing_history) > 1:
            corr = jnp.sum(forcing_history[0].conj() * forcing_history[-1]).real
            norm0 = jnp.sqrt(jnp.sum(jnp.abs(forcing_history[0]) ** 2))
            norm_current = jnp.sqrt(jnp.sum(jnp.abs(forcing_history[-1]) ** 2))

            correlation_function.append(corr / (norm0 * norm_current))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Final state
    theta = grid.ifft(state["theta_hat"]).real
    im = ax1.imshow(theta, cmap="RdBu_r", extent=[0, L, 0, L])
    ax1.set_title(f'Final θ at t={state["time"]:.2f}')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Correlation function
    times = np.arange(1, len(correlation_function) + 1) * dt
    ax2.plot(times, correlation_function, "b-", label="Measured")

    # Theoretical exponential decay
    theory = np.exp(-times / forcing.correlation_time)
    ax2.plot(times, theory, "r--", label=f"exp(-t/{forcing.correlation_time})")

    ax2.set_xlabel("Time lag")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Forcing Temporal Correlation")
    ax2.set_ylim(-0.2, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return state, correlation_function


def example_combined_forcing():
    """Demonstrate combined stochastic forcing."""
    print("\n=== Combined Stochastic Forcing Example ===")

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Create adaptive solver
    cfl_config = CFLConfig(cfl_safety=0.8, dt_max=0.01)
    solver = AdaptivegSQGSolver(grid, alpha=1.0, nu_p=1e-6, p=8, cfl_config=cfl_config)

    # Combine multiple forcing patterns
    forcing1 = ColoredNoiseForcing(
        amplitude=0.2, spectral_slope=-2.0, k_peak=5.0, k_width=3.0  # Large scale
    )

    forcing2 = WhiteNoiseForcing(amplitude=0.3, k_min=20.0, k_max=40.0)  # Small scale

    forcing3 = StochasticVortexForcing(amplitude=1.0, vortex_size=0.1, injection_rate=0.5)

    # Create combined forcing with weights
    combined_forcing = create_combined_stochastic_forcing(
        [forcing1, forcing2, forcing3], weights=[1.0, 1.0, 0.5]  # Relative weights
    )

    # Initialize
    state = solver.initialize(seed=42)

    print("Evolving with combined forcing...")

    # Storage
    energy_history = []
    spectra_history = []

    def forcing_wrapper(state, dt, key):
        return combined_forcing(state["theta_hat"], key, dt, grid)

    # Animation
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    key = jax.random.PRNGKey(999)
    for step in range(1000):
        key, subkey = jax.random.split(key)

        # Step with combined forcing
        state, info = solver.step(state, forcing=forcing_wrapper, key=subkey)

        if step % 50 == 0:
            # Compute diagnostics
            energy = compute_total_energy(state["theta_hat"], grid)
            energy_history.append((state["time"], energy))

            k_bins, spectrum = compute_energy_spectrum(state["theta_hat"], grid)
            spectra_history.append(spectrum)

            # Update plots
            # Physical space
            axes[0, 0].clear()
            theta = grid.ifft(state["theta_hat"]).real
            im = axes[0, 0].imshow(theta, cmap="RdBu_r", extent=[0, L, 0, L], vmin=-3, vmax=3)
            axes[0, 0].set_title(f'θ at t={state["time"]:.2f}')
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")

            # Energy spectrum with forcing ranges
            axes[0, 1].clear()
            axes[0, 1].loglog(k_bins, spectrum, "b-", linewidth=2)
            axes[0, 1].loglog(k_bins, k_bins ** (-5 / 3), "k--", alpha=0.5, label="k^{-5/3}")

            # Mark forcing ranges
            axes[0, 1].axvspan(3, 8, alpha=0.2, color="red", label="Large-scale forcing")
            axes[0, 1].axvspan(20, 40, alpha=0.2, color="green", label="Small-scale forcing")

            axes[0, 1].set_xlabel("k")
            axes[0, 1].set_ylabel("E(k)")
            axes[0, 1].set_title("Energy Spectrum")
            axes[0, 1].set_xlim(1, N // 2)
            axes[0, 1].set_ylim(1e-8, 1e0)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Energy history
            axes[1, 0].clear()
            if energy_history:
                times, energies = zip(*energy_history)
                axes[1, 0].plot(times, energies, "g-", linewidth=2)
            axes[1, 0].set_xlabel("Time")
            axes[1, 0].set_ylabel("Total Energy")
            axes[1, 0].set_title("Energy Evolution")
            axes[1, 0].grid(True, alpha=0.3)

            # Timestep adaptation
            axes[1, 1].clear()
            stats = solver.timestepper.get_statistics()
            if solver.timestepper.dt_history:
                axes[1, 1].semilogy(
                    solver.timestepper.time_history, solver.timestepper.dt_history, "b-", alpha=0.7
                )
                axes[1, 1].axhline(
                    stats["dt_mean"],
                    color="red",
                    linestyle="--",
                    label=f'Mean: {stats["dt_mean"]:.3e}',
                )
            axes[1, 1].set_xlabel("Time")
            axes[1, 1].set_ylabel("Timestep")
            axes[1, 1].set_title("Adaptive Timestep")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    plt.show()

    # Print statistics
    stats = solver.timestepper.get_statistics()
    print(f"\nSimulation statistics:")
    print(f"  Total steps: {stats['n_steps']}")
    print(f"  Mean timestep: {stats['dt_mean']:.3e}")
    print(f"  Mean CFL: {stats['cfl_mean']:.3f}")
    print(f"  Efficiency: {stats['efficiency']*100:.1f}%")

    return state, energy_history, spectra_history


def main():
    """Run all examples."""
    print("Stochastic Forcing Examples for pygSQuiG")
    print("========================================")

    # Choose which example to run
    examples = {
        "1": ("White Noise Forcing", example_white_noise),
        "2": ("Colored Noise Forcing", example_colored_noise),
        "3": ("Vortex Injection", example_vortex_injection),
        "4": ("Ornstein-Uhlenbeck Process", example_ou_forcing),
        "5": ("Combined Forcing", example_combined_forcing),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")

    choice = input("\nSelect example (1-5): ")

    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        func()
    else:
        print("Invalid choice. Running all examples...")
        for name, func in examples.values():
            print(f"\nRunning: {name}")
            func()
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()

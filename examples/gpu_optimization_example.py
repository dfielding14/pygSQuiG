#!/usr/bin/env python
"""
Example: GPU optimization and multi-GPU ensemble simulations.

This example demonstrates:
1. Basic GPU optimization
2. Memory optimization strategies
3. Multi-GPU ensemble runs
4. Performance benchmarking
"""

import numpy as np
import jax
import jax.numpy as jnp
import time

from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.gpu_utils import (
    setup_device,
    optimize_memory_layout,
    GPUOptimizedSolver,
    benchmark_gpu_performance,
)
from pygsquig.core.multi_gpu import MultiGPUSolver, ParallelConfig, ensemble_statistics
from pygsquig.forcing.ring_forcing import RingForcing
from pygsquig.utils.diagnostics import compute_energy_spectrum


def example_1_basic_gpu():
    """Example 1: Basic GPU optimization."""
    print("=" * 60)
    print("Example 1: Basic GPU Optimization")
    print("=" * 60)

    # Setup device
    device = setup_device("auto")  # Auto-select best device
    print(f"Using device: {device}")
    print(f"Platform: {device.platform}")

    # Create grid and solver
    N = 512
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Get memory optimization recommendations
    mem_params = optimize_memory_layout(N)
    print(f"\nMemory optimization for N={N}:")
    for key, value in mem_params.items():
        print(f"  {key}: {value}")

    # Create base solver
    base_solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-6, p=8)

    # Create GPU-optimized solver
    gpu_solver = GPUOptimizedSolver(base_solver, device=device, memory_optimization=True)

    # Initialize and run
    state = base_solver.initialize(seed=42)

    print("\nRunning simulation...")
    dt = 0.001
    n_steps = 100

    # Timing
    start = time.perf_counter()

    for i in range(n_steps):
        state = gpu_solver.step(state, dt)

        if (i + 1) % 20 == 0:
            energy = 0.5 * jnp.mean(jnp.abs(jnp.fft.ifft2(state["theta_hat"])) ** 2)
            print(f"  Step {i+1}: t={state['time']:.3f}, Energy={energy:.6f}")

    elapsed = time.perf_counter() - start
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Time per step: {elapsed/n_steps*1000:.2f} ms")

    # Memory usage (if available)
    mem_stats = gpu_solver.get_memory_usage()
    if "used_mb" in mem_stats:
        print(f"\nGPU Memory used: {mem_stats['used_mb']:.1f} MB")
        print(f"GPU Memory total: {mem_stats['total_mb']:.1f} MB")


def example_2_ensemble_simulation():
    """Example 2: Multi-GPU ensemble simulation."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-GPU Ensemble Simulation")
    print("=" * 60)

    n_devices = jax.device_count()
    print(f"Number of devices available: {n_devices}")

    if n_devices < 2:
        print("Note: Running on single device. Multi-GPU benefits require multiple devices.")

    # Setup for ensemble
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Configure parallel execution
    ensemble_size = min(n_devices, 4)  # Run 4 ensemble members or as many as devices
    config = ParallelConfig(n_devices=n_devices, ensemble_size=ensemble_size)

    # Create multi-GPU solver
    multi_solver = MultiGPUSolver(grid=grid, alpha=1.0, nu_p=1e-6, p=8, parallel_config=config)

    print(f"\nRunning ensemble with {ensemble_size} members...")

    # Run ensemble
    results = multi_solver.run_ensemble(n_steps=200, dt=0.001, save_interval=50, seed=12345)

    # Display results
    print("\nEnsemble results:")
    print("Time    Mean Energy    Std Energy")
    print("-" * 35)
    for i, t in enumerate(results["times"]):
        print(f"{t:5.2f}   {results['mean_energy'][i]:10.6f}   {results['std_energy'][i]:10.6f}")

    # Compute ensemble statistics if we saved states
    if results["states"]:
        print("\nComputing ensemble statistics...")
        stats = ensemble_statistics(results["states"][-1], grid, alpha=1.0)  # Final states
        print(
            f"Final energy range: [{stats['energy_range'][0]:.6f}, {stats['energy_range'][1]:.6f}]"
        )


def example_3_performance_comparison():
    """Example 3: Performance comparison across grid sizes."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Benchmarking")
    print("=" * 60)

    # Benchmark different grid sizes
    grid_sizes = [128, 256, 512]

    print("Benchmarking performance...")
    print("(This may take a minute)\n")

    results = benchmark_gpu_performance(grid_sizes=grid_sizes, device_type="auto")

    # Display results
    print(f"Device: {results['device']}")
    print(f"Platform: {results['platform']}")
    print("\nGrid Size   Time/Step    Throughput")
    print("-" * 40)

    for N in grid_sizes:
        time_ms = results["times"][N] * 1000
        throughput_mps = results["throughput"][N] / 1e6
        print(f"{N:^9}   {time_ms:7.2f} ms   {throughput_mps:6.2f} Mpoints/s")

    # Performance scaling
    if len(grid_sizes) > 1:
        print("\nPerformance scaling:")
        base_N = grid_sizes[0]
        base_time = results["times"][base_N]

        for N in grid_sizes[1:]:
            scaling = results["times"][N] / base_time
            expected = (N / base_N) ** 2  # O(N²) expected
            efficiency = expected / scaling
            print(
                f"  {base_N}→{N}: {scaling:.2f}x slower (expected {expected:.1f}x, efficiency {efficiency:.1%})"
            )


def example_4_forced_turbulence_gpu():
    """Example 4: Forced turbulence with GPU optimization."""
    print("\n" + "=" * 60)
    print("Example 4: GPU-Optimized Forced Turbulence")
    print("=" * 60)

    # Larger grid for GPU benefit
    N = 1024
    L = 2 * np.pi
    grid = make_grid(N, L)

    print(f"Grid size: {N}x{N}")

    # Create solver with forcing
    base_solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-8, p=8)
    gpu_solver = GPUOptimizedSolver(base_solver)

    # Setup forcing
    forcing = RingForcing(kf=30.0, dk=2.0, epsilon=0.1)

    # Initialize
    state = base_solver.initialize(seed=42)

    # Time parameters
    dt = 0.0005
    t_max = 1.0
    n_steps = int(t_max / dt)
    output_interval = n_steps // 10

    print(f"\nSimulating forced turbulence for t={t_max}")
    print(f"Time step: {dt}, Total steps: {n_steps}")

    # Storage for diagnostics
    times = []
    energies = []
    enstrophies = []

    # PRNG for forcing
    key = jax.random.PRNGKey(12345)

    # Main loop with timing
    start = time.perf_counter()

    for step in range(n_steps):
        # Split key
        key, subkey = jax.random.split(key)

        # Step with forcing
        state = gpu_solver.step(state, dt, forcing=forcing, key=subkey, grid=grid)

        # Diagnostics
        if step % output_interval == 0:
            theta = jnp.fft.ifft2(state["theta_hat"]).real
            energy = 0.5 * jnp.mean(theta**2)
            enstrophy = 0.5 * jnp.mean(jnp.abs(state["theta_hat"]) ** 2 * grid.k2)

            times.append(state["time"])
            energies.append(float(energy))
            enstrophies.append(float(enstrophy))

            elapsed = time.perf_counter() - start
            steps_per_sec = (step + 1) / elapsed
            eta = (n_steps - step - 1) / steps_per_sec

            print(
                f"Step {step:5d}: t={state['time']:6.3f}, E={energy:.4f}, "
                f"Z={enstrophy:.2f}, Speed: {steps_per_sec:.1f} steps/s, ETA: {eta:.1f}s"
            )

    total_time = time.perf_counter() - start

    print(f"\nSimulation complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per step: {total_time/n_steps*1000:.2f} ms")
    print(f"Effective resolution updates per second: {N*N*n_steps/total_time/1e6:.2f} Mpoints/s")

    # Memory usage
    mem_stats = gpu_solver.get_memory_usage()
    if "used_mb" in mem_stats:
        print(f"\nPeak GPU memory usage: {mem_stats['used_mb']:.1f} MB")


def main():
    """Run all examples."""
    print("GPU Optimization Examples for pygSQuiG")
    print("=" * 60)

    # Show JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Run examples
    example_1_basic_gpu()

    if jax.device_count() > 1:
        example_2_ensemble_simulation()
    else:
        print("\n[Skipping Example 2: Multi-GPU ensemble requires multiple devices]")

    example_3_performance_comparison()
    example_4_forced_turbulence_gpu()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("\nKey takeaways:")
    print("- GPU optimization can significantly speed up large simulations")
    print("- Memory optimization is crucial for large grids")
    print("- Multi-GPU enables efficient ensemble simulations")
    print("- Performance scales well with problem size on GPU")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Benchmark script to measure JIT compilation performance improvement.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from pygsquig.core.grid import make_grid
from pygsquig.core.operators import compute_velocity_from_theta, gradient, jacobian
from pygsquig.core.solver import gSQGSolver


def benchmark_operator(name: str, func, *args, n_warmup: int = 3, n_runs: int = 10):
    """Benchmark a single operator function."""
    # Warmup (for JIT compilation)
    for _ in range(n_warmup):
        _ = func(*args)

    # Block until computation completes
    jax.block_until_ready(func(*args))

    # Timing runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        jax.block_until_ready(result)  # Wait for GPU computation
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times[1:])  # Skip first run
    std_time = np.std(times[1:])

    print(f"{name:30s}: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    return mean_time


def benchmark_solver_step(solver, state, dt, n_warmup=3, n_runs=10):
    """Benchmark a full solver step."""
    # Warmup
    for _ in range(n_warmup):
        state = solver.step(state, dt)

    # Timing runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        state = solver.step(state, dt)
        # Extract array to force computation
        jax.block_until_ready(state["theta_hat"])
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times[1:])
    std_time = np.std(times[1:])

    print(f"{'Full solver step':30s}: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    return mean_time


def main():
    print("=" * 60)
    print("JAX JIT Compilation Performance Benchmark")
    print("=" * 60)

    # Setup
    N = 256
    L = 2 * jnp.pi
    alpha = 1.0

    print(f"\nGrid size: {N}×{N}")
    print(f"Platform: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print()

    # Create grid and solver
    grid = make_grid(N, L)
    solver = gSQGSolver(grid, alpha=alpha, nu_p=1e-8, p=8)

    # Initialize state
    theta0 = jnp.sin(4 * grid.x) * jnp.cos(4 * grid.y)
    state = solver.initialize(theta0)
    theta_hat = state["theta_hat"]

    # Create test data
    theta = jnp.sin(grid.x) * jnp.cos(grid.y)
    u = jnp.ones_like(theta) * 0.1
    v = jnp.ones_like(theta) * 0.1

    print("Benchmarking individual operators:")
    print("-" * 60)

    # Benchmark individual operators
    times = {}

    # Gradient
    times["gradient"] = benchmark_operator("gradient", gradient, theta_hat, grid)

    # Velocity computation
    times["velocity"] = benchmark_operator(
        "compute_velocity_from_theta", compute_velocity_from_theta, theta_hat, grid, alpha
    )

    # Jacobian (advection)
    times["jacobian"] = benchmark_operator("jacobian", jacobian, theta, u, v, grid)

    print("\nBenchmarking full simulation step:")
    print("-" * 60)

    # Full solver step
    dt = 0.001
    times["solver_step"] = benchmark_solver_step(solver, state, dt)

    # Performance analysis
    print("\nPerformance Analysis:")
    print("-" * 60)

    # Estimate steps per second
    steps_per_second = 1.0 / times["solver_step"]
    print(f"Steps per second: {steps_per_second:.1f}")

    # Time for 10k steps
    time_10k = 10000 * times["solver_step"]
    print(f"Time for 10k steps: {time_10k:.1f} seconds ({time_10k/60:.1f} minutes)")

    # Breakdown
    print("\nOperator breakdown (% of solver step):")
    print(f"  Velocity computation: {times['velocity']/times['solver_step']*100:.1f}%")
    print(f"  Jacobian (advection): {times['jacobian']/times['solver_step']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance profiling script for pygSQuiG solver.

This script profiles the computational performance of different solver components
to identify bottlenecks and optimization opportunities.
"""

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from pygsquig.core.grid import fft2, ifft2, make_grid
from pygsquig.core.operators import (
    compute_velocity_from_theta,
    fractional_laplacian,
    gradient,
    jacobian,
    laplacian,
)
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing.damping import CombinedDamping
from pygsquig.forcing.ring_forcing import RingForcing


class PerformanceProfiler:
    """Profile computational performance of pygSQuiG components."""

    def __init__(self, N=256, L=2 * np.pi):
        """Initialize profiler with given resolution."""
        self.N = N
        self.L = L
        self.grid = make_grid(N, L)
        self.solver = gSQGSolver(self.grid, alpha=1.0, nu_p=1e-4, p=8)

        # Initialize test data
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, shape=(N, N))
        self.theta_hat = fft2(theta)
        self.theta = theta

        # Warmup JIT compilation
        self._warmup()

    def _warmup(self):
        """Warmup JIT compilation for all functions."""
        print("Warming up JIT compilation...")

        # Core operations
        _ = compute_velocity_from_theta(self.theta_hat, self.grid, 1.0)
        _ = jacobian(self.theta, self.theta, self.theta, self.grid)
        _ = gradient(self.theta_hat, self.grid)
        _ = laplacian(self.theta_hat, self.grid)
        _ = fractional_laplacian(self.theta_hat, self.grid, 1.0)

        # Solver step
        state = self.solver.initialize(seed=42)
        _ = self.solver.step(state, 0.001)

        print("JIT warmup complete!\n")

    def time_operation(self, func, *args, n_runs=100, **kwargs):
        """Time a function execution."""
        # Ensure function is compiled
        _ = func(*args, **kwargs)

        # Time multiple runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            # Handle both single arrays and tuples of arrays
            if isinstance(result, tuple):
                for arr in result:
                    if hasattr(arr, "block_until_ready"):
                        arr.block_until_ready()
            elif hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return np.mean(times[10:]), np.std(times[10:])  # Skip first few for stability

    def profile_fft_operations(self):
        """Profile FFT operations."""
        print("=== FFT Operations ===")

        # FFT forward
        mean_fft, std_fft = self.time_operation(fft2, self.theta)
        print(f"FFT2 forward:  {mean_fft*1000:.3f} ± {std_fft*1000:.3f} ms")

        # FFT inverse
        mean_ifft, std_ifft = self.time_operation(ifft2, self.theta_hat)
        print(f"IFFT2 inverse: {mean_ifft*1000:.3f} ± {std_ifft*1000:.3f} ms")

        return {"fft": mean_fft, "ifft": mean_ifft}

    def profile_operators(self):
        """Profile spectral operators."""
        print("\n=== Spectral Operators ===")

        results = {}

        # Gradient
        mean, std = self.time_operation(gradient, self.theta_hat, self.grid)
        print(f"Gradient:              {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["gradient"] = mean

        # Laplacian
        mean, std = self.time_operation(laplacian, self.theta_hat, self.grid)
        print(f"Laplacian:             {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["laplacian"] = mean

        # Fractional Laplacian
        mean, std = self.time_operation(fractional_laplacian, self.theta_hat, self.grid, 1.0)
        print(f"Fractional Laplacian:  {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["frac_laplacian"] = mean

        # Velocity computation
        mean, std = self.time_operation(
            compute_velocity_from_theta, self.theta_hat, self.grid, 1.0
        )
        print(f"Velocity from theta:   {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["velocity"] = mean

        # Jacobian (most expensive)
        u = jax.random.normal(jax.random.PRNGKey(1), shape=(self.N, self.N))
        v = jax.random.normal(jax.random.PRNGKey(2), shape=(self.N, self.N))
        mean, std = self.time_operation(jacobian, self.theta, u, v, self.grid)
        print(f"Jacobian:              {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["jacobian"] = mean

        return results

    def profile_solver_components(self):
        """Profile solver components."""
        print("\n=== Solver Components ===")

        state = self.solver.initialize(seed=42)
        results = {}

        # RHS computation (without forcing/damping)
        mean, std = self.time_operation(self.solver.compute_rhs, state["theta_hat"])
        print(f"RHS computation:       {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["rhs"] = mean

        # RHS with forcing/damping
        forcing = RingForcing(kf=30.0, dk=2.0, epsilon=0.1)
        damping = CombinedDamping(mu=0.1, kf=30.0, nu_p=1e-8, p=8)
        key = jax.random.PRNGKey(123)

        # Define a wrapper that properly calls forcing and damping
        def compute_rhs_with_forcing(theta_hat):
            rhs = self.solver.compute_rhs(theta_hat)
            forcing_hat = forcing(theta_hat, key, 0.001, self.grid)
            damping_hat = damping(theta_hat, self.grid)
            return rhs + forcing_hat + damping_hat

        mean, std = self.time_operation(compute_rhs_with_forcing, state["theta_hat"])
        print(f"RHS with F+D:          {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["rhs_forced"] = mean

        # Full time step
        mean, std = self.time_operation(self.solver.step, state, 0.001)
        print(f"Full solver step:      {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["step"] = mean

        # With forcing and damping (reuse from above)
        # Create wrapped forcing/damping functions
        forcing_fn = lambda theta_hat: forcing(theta_hat, key, 0.001, self.grid)
        damping_fn = lambda theta_hat: damping(theta_hat, self.grid)

        mean, std = self.time_operation(
            self.solver.step, state, 0.001, forcing=forcing_fn, damping=damping_fn
        )
        print(f"Step with F+D:         {mean*1000:.3f} ± {std*1000:.3f} ms")
        results["step_forced"] = mean

        return results

    def profile_resolution_scaling(self):
        """Profile performance scaling with resolution."""
        print("\n=== Resolution Scaling ===")

        resolutions = [64, 128, 256, 512]
        step_times = []
        fft_times = []
        jacobian_times = []

        for N in resolutions:
            print(f"\nN = {N}:")

            # Create solver for this resolution
            grid = make_grid(N, self.L)
            solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-4, p=8)
            state = solver.initialize(seed=42)

            # Warmup
            for _ in range(5):
                state = solver.step(state, 0.001)

            # Time full step
            mean_step, _ = self.time_operation(solver.step, state, 0.001, n_runs=50)
            step_times.append(mean_step)
            print(f"  Step time: {mean_step*1000:.2f} ms ({1/mean_step:.1f} steps/sec)")

            # Time FFT
            theta = jax.random.normal(jax.random.PRNGKey(42), shape=(N, N))
            mean_fft, _ = self.time_operation(fft2, theta, n_runs=50)
            fft_times.append(mean_fft)
            print(f"  FFT time:  {mean_fft*1000:.2f} ms")

            # Time Jacobian
            u = jax.random.normal(jax.random.PRNGKey(1), shape=(N, N))
            v = jax.random.normal(jax.random.PRNGKey(2), shape=(N, N))
            mean_jac, _ = self.time_operation(jacobian, theta, u, v, grid, n_runs=20)
            jacobian_times.append(mean_jac)
            print(f"  Jacobian:  {mean_jac*1000:.2f} ms")

        return resolutions, step_times, fft_times, jacobian_times

    def create_performance_report(self):
        """Create comprehensive performance report."""
        print("\n" + "=" * 60)
        print("pygSQuiG Performance Profile Report")
        print(f"Resolution: {self.N}×{self.N}, Domain: {self.L}×{self.L}")
        print("=" * 60)

        # Profile all components
        fft_results = self.profile_fft_operations()
        operator_results = self.profile_operators()
        solver_results = self.profile_solver_components()

        # Resolution scaling
        resolutions, step_times, fft_times, jacobian_times = self.profile_resolution_scaling()

        # Create plots
        self.plot_results(
            fft_results,
            operator_results,
            solver_results,
            resolutions,
            step_times,
            fft_times,
            jacobian_times,
        )

        # Summary
        print("\n=== Performance Summary ===")
        total_step_time = solver_results["step"]
        print(f"Time per step: {total_step_time*1000:.2f} ms")
        print(f"Steps per second: {1/total_step_time:.1f}")
        print(f"Simulation speed: {self.N**2 / total_step_time / 1e6:.1f} Mgridpoints/sec")

        # Breakdown
        rhs_time = solver_results["rhs"]
        overhead = total_step_time - rhs_time
        print("\nTime breakdown:")
        print(f"  RHS computation: {rhs_time/total_step_time*100:.1f}%")
        print(f"  RK4 overhead:    {overhead/total_step_time*100:.1f}%")

        # Identify bottlenecks
        print("\nMain bottlenecks:")
        print(
            f"  Jacobian:     {operator_results['jacobian']*1000:.2f} ms ({operator_results['jacobian']/rhs_time*100:.0f}% of RHS)"
        )
        print(f"  FFT/IFFT:     {(fft_results['fft']+fft_results['ifft'])*1000:.2f} ms total")
        print(f"  Velocity:     {operator_results['velocity']*1000:.2f} ms")

    def plot_results(
        self,
        fft_results,
        operator_results,
        solver_results,
        resolutions,
        step_times,
        fft_times,
        jacobian_times,
    ):
        """Create performance visualization plots."""
        output_dir = Path("performance_analysis")
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Component breakdown
        ax = axes[0, 0]
        components = list(operator_results.keys())
        times = [operator_results[c] * 1000 for c in components]
        ax.bar(components, times)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Operator Performance")
        ax.tick_params(axis="x", rotation=45)

        # Resolution scaling
        ax = axes[0, 1]
        ax.loglog(resolutions, step_times, "bo-", label="Total step", linewidth=2, markersize=8)
        ax.loglog(resolutions, fft_times, "rs-", label="FFT", linewidth=2, markersize=6)
        ax.loglog(resolutions, jacobian_times, "g^-", label="Jacobian", linewidth=2, markersize=6)

        # Add O(N²) reference
        N_ref = np.array(resolutions)
        ax.loglog(
            N_ref, step_times[0] * (N_ref / resolutions[0]) ** 2, "k--", alpha=0.5, label="O(N²)"
        )

        ax.set_xlabel("Resolution N")
        ax.set_ylabel("Time (s)")
        ax.set_title("Computational Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        # Steps per second vs resolution
        ax = axes[1, 0]
        steps_per_sec = [1 / t for t in step_times]
        ax.loglog(resolutions, steps_per_sec, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Resolution N")
        ax.set_ylabel("Steps/second")
        ax.set_title("Simulation Speed")
        ax.grid(True, alpha=0.3, which="both")

        # Time breakdown pie chart
        ax = axes[1, 1]
        labels = ["Jacobian", "Velocity", "FFT ops", "Other"]
        sizes = [
            operator_results["jacobian"],
            operator_results["velocity"],
            fft_results["fft"] + fft_results["ifft"],
            solver_results["rhs"]
            - operator_results["jacobian"]
            - operator_results["velocity"]
            - fft_results["fft"]
            - fft_results["ifft"],
        ]
        sizes = [max(0, s) for s in sizes]  # Ensure non-negative
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("RHS Computation Breakdown")

        plt.tight_layout()
        plt.savefig(output_dir / "performance_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\nPerformance plots saved to {output_dir / 'performance_analysis.png'}")
        plt.close()


def suggest_optimizations(profiler):
    """Suggest potential optimizations based on profiling results."""
    print("\n" + "=" * 60)
    print("Optimization Suggestions")
    print("=" * 60)

    print("\n1. **Jacobian Optimization**:")
    print("   - The Jacobian computation is the main bottleneck")
    print("   - Consider caching FFT plans for repeated transforms")
    print("   - Explore fused operations to reduce memory traffic")

    print("\n2. **Memory Layout**:")
    print("   - Ensure arrays are contiguous in memory")
    print("   - Consider batching multiple FFTs together")

    print("\n3. **GPU-Specific**:")
    print("   - Use GPU if available: export JAX_PLATFORM_NAME=gpu")
    print("   - Profile with larger grids where GPU advantage is clearer")
    print("   - Consider using jax.pmap for multi-GPU scaling")

    print("\n4. **Algorithm Improvements**:")
    print("   - Implement adaptive timestepping to use larger dt")
    print("   - Use lower-order time integration for less accuracy-critical runs")
    print("   - Consider implicit-explicit (IMEX) schemes for stiff dissipation")

    print("\n5. **JAX-Specific**:")
    print("   - Ensure all operations are JIT-compiled")
    print("   - Minimize Python overhead in hot loops")
    print("   - Use lax.scan for time-stepping loops")


def main():
    """Run performance profiling."""
    # Check if GPU is available
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print()

    # Create profiler with typical resolution
    profiler = PerformanceProfiler(N=256)

    # Run comprehensive profiling
    profiler.create_performance_report()

    # Provide optimization suggestions
    suggest_optimizations(profiler)


if __name__ == "__main__":
    main()

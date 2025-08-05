"""
Multi-GPU support for pygSQuiG using JAX's parallel primitives.

This module provides domain decomposition and ensemble parallelism
across multiple GPUs using pmap and shard_map.
"""

import functools
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.exceptions import ConfigurationError


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    n_devices: int
    ensemble_size: Optional[int] = None
    domain_decomposition: bool = False
    axis_names: tuple[str, ...] = ("device",)
    mesh_shape: Optional[tuple[int, ...]] = None


def create_device_mesh(
    n_devices: Optional[int] = None, mesh_shape: Optional[tuple[int, ...]] = None
) -> Mesh:
    """Create device mesh for multi-GPU execution.

    Args:
        n_devices: Number of devices (defaults to all available)
        mesh_shape: Shape of device mesh (e.g., (2, 2) for 4 GPUs)

    Returns:
        JAX Mesh object
    """
    if n_devices is None:
        n_devices = jax.device_count()

    if mesh_shape is None:
        # Default to 1D mesh
        mesh_shape = (n_devices,)
    else:
        # Verify mesh shape matches device count
        if np.prod(mesh_shape) != n_devices:
            raise ConfigurationError(
                f"Mesh shape {mesh_shape} incompatible with {n_devices} devices"
            )

    devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices, axis_names=("device",) if len(mesh_shape) == 1 else ("x", "y"))


class MultiGPUSolver:
    """Multi-GPU parallel solver for gSQG equations.

    Supports both ensemble parallelism and domain decomposition.
    """

    def __init__(
        self,
        grid: Grid,
        alpha: float,
        nu_p: float = 0.0,
        p: int = 8,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize multi-GPU solver.

        Args:
            grid: Grid object
            alpha: Fractional power
            nu_p: Hyperviscosity coefficient
            p: Hyperviscosity order
            parallel_config: Parallel execution configuration
        """
        self.grid = grid
        self.alpha = alpha
        self.nu_p = nu_p
        self.p = p

        # Setup parallel configuration
        if parallel_config is None:
            self.config = ParallelConfig(n_devices=jax.device_count())
        else:
            self.config = parallel_config

        # Create device mesh
        self.mesh = create_device_mesh(self.config.n_devices, self.config.mesh_shape)

        # Setup parallel functions
        self._setup_parallel_functions()

    def _setup_parallel_functions(self):
        """Create parallelized versions of core functions."""

        if self.config.ensemble_size is not None:
            # Ensemble parallelism - each device runs independent simulation
            self._setup_ensemble_parallel()
        elif self.config.domain_decomposition:
            # Domain decomposition - distribute grid across devices
            self._setup_domain_decomposition()
        else:
            # Simple data parallelism
            self._setup_data_parallel()

    def _setup_ensemble_parallel(self):
        """Setup functions for ensemble parallelism."""
        # Each device handles one ensemble member

        @functools.partial(jax.pmap, axis_name="ensemble")
        def ensemble_step(theta_hat, dt, key):
            """Single step for ensemble member."""
            from pygsquig.core.solver import _compute_core_rhs
            from pygsquig.core.time_integrator import rk4_step

            # Each device computes independently
            def rhs_fn(theta):
                return _compute_core_rhs(theta, self.grid, self.alpha, self.nu_p, self.p)

            # RK4 step
            theta_new = rk4_step(theta_hat, rhs_fn, dt)

            return theta_new

        self.ensemble_step = ensemble_step

        # Initialize ensemble states
        @functools.partial(jax.pmap, axis_name="ensemble", static_broadcasted_argnums=(1,))
        def init_ensemble(keys, shape):
            """Initialize ensemble with different random seeds."""
            # Each device gets different key
            theta = jax.random.normal(keys, shape, dtype=jnp.float64)
            theta_hat = fft2(theta)

            # Apply smoothing
            k_mag = jnp.sqrt(self.grid.k2)
            k_cutoff = self.grid.N // 4 * 2 * jnp.pi / self.grid.L
            mask = k_mag < k_cutoff
            theta_hat = theta_hat * mask

            return theta_hat

        self.init_ensemble = init_ensemble

    def _setup_domain_decomposition(self):
        """Setup domain decomposition across devices."""
        # This is more complex - need to handle halo exchanges
        # For FFT-based methods, full implementation requires careful design

        # Simplified version - decompose in physical space operations only
        N = self.grid.N
        n_dev = self.config.n_devices

        if N % n_dev != 0:
            raise ConfigurationError(f"Grid size {N} must be divisible by device count {n_dev}")

        N // n_dev

        @functools.partial(
            shard_map, mesh=self.mesh, in_specs=P("device", None), out_specs=P("device", None)
        )
        def local_jacobian(theta_local, u_local, v_local):
            """Compute Jacobian on local subdomain."""
            # This is simplified - real implementation needs halo exchange
            from pygsquig.core.operators import jacobian

            # Compute local contribution
            # Note: This is incomplete without proper boundary handling
            return jacobian(theta_local, u_local, v_local, self.grid)

        self.local_jacobian = local_jacobian

    def _setup_data_parallel(self):
        """Setup simple data parallelism."""
        # Process multiple states in parallel

        @functools.partial(
            jax.pmap, axis_name="batch", static_broadcasted_argnums=(2,)  # dt is static
        )
        def batch_step(theta_hat_batch, time_batch, dt):
            """Step multiple states in parallel."""
            from pygsquig.core.solver import _compute_core_rhs
            from pygsquig.core.time_integrator import rk4_step

            def rhs_fn(theta):
                return _compute_core_rhs(theta, self.grid, self.alpha, self.nu_p, self.p)

            theta_new = rk4_step(theta_hat_batch, rhs_fn, dt)
            time_new = time_batch + dt

            return theta_new, time_new

        self.batch_step = batch_step

    def run_ensemble(
        self, n_steps: int, dt: float, save_interval: int = 100, seed: int = 42
    ) -> dict[str, Any]:
        """Run ensemble simulation across GPUs.

        Args:
            n_steps: Number of time steps
            dt: Time step size
            save_interval: Steps between saves
            seed: Random seed for initialization

        Returns:
            Dictionary with ensemble results
        """
        if self.config.ensemble_size is None:
            raise ConfigurationError("Ensemble size not specified")

        # Generate keys for each ensemble member
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, self.config.ensemble_size)

        # Initialize ensemble
        theta_hat_ensemble = self.init_ensemble(keys, (self.grid.N, self.grid.N))

        # Storage for results
        results: dict[str, list[Any]] = {
            "times": [],
            "mean_energy": [],
            "std_energy": [],
            "states": [],
        }

        # Time evolution
        t = 0.0
        for step in range(n_steps):
            # Split keys for stochastic forcing if needed
            keys = jax.random.split(keys[0], self.config.ensemble_size)

            # Parallel step
            theta_hat_ensemble = self.ensemble_step(theta_hat_ensemble, dt, keys)
            t += dt

            # Save diagnostics
            if step % save_interval == 0:
                # Compute energy for each member
                theta_ensemble = jax.vmap(ifft2)(theta_hat_ensemble)
                energy_ensemble = jax.vmap(lambda theta: 0.5 * jnp.mean(theta**2))(theta_ensemble)

                results["times"].append(t)
                results["mean_energy"].append(float(jnp.mean(energy_ensemble)))
                results["std_energy"].append(float(jnp.std(energy_ensemble)))

                # Optionally save full states
                if step % (save_interval * 10) == 0:
                    results["states"].append(theta_hat_ensemble)

        return results

    def benchmark_scaling(self, n_steps: int = 100, dt: float = 0.001) -> dict[str, float]:
        """Benchmark multi-GPU scaling.

        Args:
            n_steps: Number of steps to benchmark
            dt: Time step size

        Returns:
            Dictionary with timing results
        """
        import time

        results = {}

        # Single device baseline
        if self.config.n_devices > 1:
            # Time single device
            single_solver = MultiGPUSolver(
                self.grid, self.alpha, self.nu_p, self.p, ParallelConfig(n_devices=1)
            )

            key = jax.random.PRNGKey(42)
            theta_hat = single_solver.init_ensemble(key[None, ...], (self.grid.N, self.grid.N))[0]

            start = time.perf_counter()
            for _ in range(n_steps):
                theta_hat = single_solver.ensemble_step(theta_hat[None, ...], dt, key[None, ...])[
                    0
                ]
            jax.block_until_ready(theta_hat)
            single_time = time.perf_counter() - start

            results["single_device_time"] = single_time

        # Multi-device timing
        if self.config.ensemble_size:
            # Ensemble parallel
            ensemble_results = self.run_ensemble(n_steps, dt, save_interval=n_steps + 1)
            # Extract time from run
            results["ensemble_parallel_time"] = ensemble_results.get("total_time", 0)
        else:
            # Batch parallel
            n_batch = self.config.n_devices
            keys = jax.random.split(jax.random.PRNGKey(42), n_batch)
            theta_hat_batch = jax.vmap(
                lambda k: fft2(jax.random.normal(k, (self.grid.N, self.grid.N)))
            )(keys)
            time_batch = jnp.zeros(n_batch)

            start = time.perf_counter()
            for _ in range(n_steps):
                theta_hat_batch, time_batch = self.batch_step(theta_hat_batch, time_batch, dt)
            jax.block_until_ready(theta_hat_batch)
            multi_time = time.perf_counter() - start

            results["multi_device_time"] = multi_time

            if "single_device_time" in results:
                results["speedup"] = results["single_device_time"] / multi_time
                results["efficiency"] = results["speedup"] / self.config.n_devices

        return results


def optimize_domain_decomposition(
    N: int, n_devices: int, decomp_type: str = "1d"
) -> dict[str, Any]:
    """Determine optimal domain decomposition strategy.

    Args:
        N: Grid size
        n_devices: Number of devices
        decomp_type: '1d' or '2d' decomposition

    Returns:
        Dictionary with decomposition parameters
    """
    if decomp_type == "1d":
        # Decompose along one dimension
        if N % n_devices != 0:
            # Find closest divisor
            divisors = [d for d in range(1, n_devices + 1) if N % d == 0]
            if divisors:
                n_devices_actual = max(d for d in divisors if d <= n_devices)
            else:
                raise ConfigurationError(
                    f"Cannot decompose grid size {N} across {n_devices} devices"
                )
        else:
            n_devices_actual = n_devices

        return {
            "type": "1d",
            "n_devices_used": n_devices_actual,
            "local_size": N // n_devices_actual,
            "halo_size": 2,  # For gradient computations
            "mesh_shape": (n_devices_actual,),
        }

    elif decomp_type == "2d":
        # 2D decomposition
        # Find factorization closest to square
        factors = []
        for i in range(1, int(np.sqrt(n_devices)) + 1):
            if n_devices % i == 0:
                factors.append((i, n_devices // i))

        if not factors:
            raise ConfigurationError(f"Cannot factor {n_devices} for 2D decomposition")

        # Choose most square-like decomposition
        px, py = min(factors, key=lambda f: abs(f[0] - f[1]))

        if N % px != 0 or N % py != 0:
            raise ConfigurationError(f"Grid size {N} incompatible with {px}x{py} decomposition")

        return {
            "type": "2d",
            "n_devices_used": n_devices,
            "local_size": (N // px, N // py),
            "halo_size": 2,
            "mesh_shape": (px, py),
            "px": px,
            "py": py,
        }

    else:
        raise ValueError(f"Unknown decomposition type: {decomp_type}")


# Helper function for ensemble statistics
def ensemble_statistics(
    ensemble_states: list[Array], grid: Grid, alpha: float
) -> dict[str, Array]:
    """Compute statistics across ensemble.

    Args:
        ensemble_states: List of states from ensemble
        grid: Grid object
        alpha: Fractional power

    Returns:
        Dictionary with ensemble statistics
    """
    from pygsquig.utils.diagnostics import compute_energy_spectrum, compute_total_energy

    # Stack states
    states_array = jnp.stack(ensemble_states)

    # Compute spectra for each member
    spectra = jax.vmap(lambda state: compute_energy_spectrum(state, grid, alpha))(states_array)

    # Mean and std of spectra
    mean_spectrum = jnp.mean(spectra, axis=0)
    std_spectrum = jnp.std(spectra, axis=0)

    # Energy statistics
    energies = jax.vmap(lambda state: compute_total_energy(state, grid, alpha))(states_array)

    return {
        "mean_spectrum": mean_spectrum,
        "std_spectrum": std_spectrum,
        "mean_energy": jnp.mean(energies),
        "std_energy": jnp.std(energies),
        "energy_range": (jnp.min(energies), jnp.max(energies)),
    }

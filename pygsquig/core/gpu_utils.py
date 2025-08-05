"""
GPU optimization utilities for pygSQuiG.

This module provides device management, memory optimization,
and multi-GPU support for the solver.
"""

# Simple logger setup
import logging
from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from pygsquig.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def get_available_devices(backend: Optional[str] = None) -> list[jax.Device]:
    """Get list of available JAX devices.

    Args:
        backend: Specific backend ('cpu', 'gpu', 'tpu') or None for all

    Returns:
        List of available devices
    """
    if backend is None:
        return list(jax.devices())
    else:
        try:
            devices = jax.devices(backend)
            return list(devices) if devices else []
        except (RuntimeError, KeyError, ValueError):
            return []


def setup_device(
    device_type: str = "auto",
    device_id: Optional[int] = None,
    memory_fraction: Optional[float] = None,
) -> jax.Device:
    """Setup JAX computation device.

    Args:
        device_type: 'cpu', 'gpu', 'tpu', or 'auto'
        device_id: Specific device ID to use
        memory_fraction: GPU memory fraction to allocate (0-1)

    Returns:
        Selected JAX device

    Raises:
        ConfigurationError: If requested device not available
    """
    # Auto-select best available device
    if device_type == "auto":
        for backend in ["gpu", "tpu", "cpu"]:
            devices = get_available_devices(backend)
            if devices:
                device_type = backend
                break
        else:
            device_type = "cpu"

    # Get devices of requested type
    devices = get_available_devices(device_type)
    if not devices:
        raise ConfigurationError(f"No {device_type} devices available")

    # Select specific device
    if device_id is not None:
        if device_id >= len(devices):
            raise ConfigurationError(
                f"Device {device_id} not found. Only {len(devices)} {device_type} devices available"
            )
        device = devices[device_id]
    else:
        device = devices[0]

    # Configure GPU memory if requested
    if device_type == "gpu" and memory_fraction is not None:
        if not 0 < memory_fraction <= 1:
            raise ConfigurationError(f"memory_fraction must be in (0, 1], got {memory_fraction}")
        # Set XLA flag for memory allocation
        import os

        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    logger.info(f"Using device: {device}")
    return device


def device_put_with_sharding(
    array: Union[np.ndarray, jnp.ndarray], device: jax.Device, sharding: Optional[Any] = None
) -> Array:
    """Put array on device with optional sharding.

    Args:
        array: Array to transfer
        device: Target device
        sharding: Optional sharding specification

    Returns:
        Array on specified device
    """
    if sharding is not None:
        return jax.device_put(array, sharding)
    else:
        return jax.device_put(array, device)


def optimize_memory_layout(N: int) -> dict[str, Any]:
    """Get optimal memory layout parameters for given grid size.

    Args:
        N: Grid size (number of points per dimension)

    Returns:
        Dictionary with memory optimization parameters
    """
    # Determine optimal batch size for operations
    # Based on typical GPU memory constraints
    total_elements = N * N
    complex_size = 16  # bytes for complex128
    array_size_mb = total_elements * complex_size / (1024 * 1024)

    # Recommendations based on array size
    recommendations: dict[str, Any] = {
        "use_float32": array_size_mb > 500,  # Use float32 for large arrays (lowered threshold)
        "chunk_size": min(N, 512),  # Optimal chunk size for operations
        "prefetch": array_size_mb < 100,  # Prefetch small arrays
        "persist_on_device": array_size_mb < 500,  # Keep medium arrays on device
    }

    # FFT optimization
    if N > 1024:
        recommendations["fft_planning"] = "measure"  # Use FFTW planning
    else:
        recommendations["fft_planning"] = "estimate"

    return recommendations


def create_sharded_grid(
    N: int, L: float, n_devices: Optional[int] = None, axis_name: str = "batch"
) -> tuple[Any, Mesh]:
    """Create grid arrays sharded across devices.

    Args:
        N: Grid size
        L: Domain size
        n_devices: Number of devices to shard across
        axis_name: Name for pmap axis

    Returns:
        Tuple of (sharded_grid, mesh)
    """
    if n_devices is None:
        n_devices = len(jax.devices())

    # Create mesh for multi-device
    device_mesh = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(device_mesh, axis_names=(axis_name,))

    # Import Grid here to avoid circular imports
    from pygsquig.core.grid import make_grid

    # Create grid on each device
    with mesh:
        # For now, replicate grid on all devices
        # In future, could shard grid for very large problems
        grid = make_grid(N, L)

        # Shard arrays across devices (replicated for now)
        sharding = jax.sharding.NamedSharding(mesh, P())

        # Put grid arrays on devices
        grid_sharded = type(grid)(
            N=grid.N,
            L=grid.L,
            x=device_put_with_sharding(grid.x, jax.devices()[0], sharding),
            y=device_put_with_sharding(grid.y, jax.devices()[0], sharding),
            kx=device_put_with_sharding(grid.kx, jax.devices()[0], sharding),
            ky=device_put_with_sharding(grid.ky, jax.devices()[0], sharding),
            k2=device_put_with_sharding(grid.k2, jax.devices()[0], sharding),
            dealias_mask=device_put_with_sharding(grid.dealias_mask, jax.devices()[0], sharding),
        )

    return grid_sharded, mesh


class GPUOptimizedSolver:
    """Wrapper for gSQGSolver with GPU optimizations.

    This class provides memory-efficient operations and multi-GPU support.
    """

    def __init__(
        self,
        base_solver,
        device: Optional[jax.Device] = None,
        enable_multi_gpu: bool = False,
        memory_optimization: bool = True,
    ):
        """Initialize GPU-optimized solver.

        Args:
            base_solver: Base gSQGSolver instance
            device: Specific device to use
            enable_multi_gpu: Enable multi-GPU parallelism
            memory_optimization: Enable memory optimization strategies
        """
        self.base_solver = base_solver
        self.device = device or jax.devices()[0]
        self.enable_multi_gpu = enable_multi_gpu
        self.memory_optimization = memory_optimization

        # Get memory optimization parameters
        if memory_optimization:
            self.mem_params = optimize_memory_layout(base_solver.grid.N)
            logger.info(f"Memory optimization: {self.mem_params}")

        # Setup multi-GPU if requested
        if enable_multi_gpu:
            self.n_devices = len(jax.devices())
            if self.n_devices > 1:
                logger.info(f"Multi-GPU enabled with {self.n_devices} devices")
                self._setup_multi_gpu()
            else:
                logger.warning("Multi-GPU requested but only 1 device available")
                self.enable_multi_gpu = False

    def _setup_multi_gpu(self):
        """Setup multi-GPU computation."""
        # Create sharded grid
        self.sharded_grid, self.mesh = create_sharded_grid(
            self.base_solver.grid.N, self.base_solver.grid.L, self.n_devices
        )

        # Create pmap versions of key functions
        self._create_parallel_functions()

    def _create_parallel_functions(self):
        """Create pmap versions of computational kernels."""
        from pygsquig.core.operators import compute_velocity_from_theta

        # Parallel velocity computation
        self.compute_velocity_parallel = jax.pmap(
            compute_velocity_from_theta,
            axis_name="device",
            static_broadcasted_argnums=(1, 2),  # grid and alpha
        )

        # Parallel RHS computation
        def compute_rhs_single(theta_hat, grid, alpha, nu_p, p):
            return self.base_solver.compute_rhs(theta_hat, forcing=None, damping=None)

        self.compute_rhs_parallel = jax.pmap(
            compute_rhs_single, axis_name="device", static_broadcasted_argnums=(1, 2, 3, 4)
        )

    def step(self, state: dict[str, Any], dt: float, **kwargs) -> dict[str, Any]:
        """Perform optimized time step.

        Args:
            state: Current state
            dt: Time step
            **kwargs: Additional arguments

        Returns:
            Updated state
        """
        if self.enable_multi_gpu and self.n_devices > 1:
            return self._step_multi_gpu(state, dt, **kwargs)
        else:
            return self._step_single_gpu(state, dt, **kwargs)

    def _step_single_gpu(self, state: dict[str, Any], dt: float, **kwargs) -> dict[str, Any]:
        """Single GPU optimized step."""
        # Ensure data is on correct device
        theta_hat = jax.device_put(state["theta_hat"], self.device)

        # Use base solver with device-local data
        state_device = {"theta_hat": theta_hat, "time": state["time"], "step": state["step"]}

        # Perform step
        new_state = self.base_solver.step(state_device, dt, **kwargs)

        return new_state

    def _step_multi_gpu(self, state: dict[str, Any], dt: float, **kwargs) -> dict[str, Any]:
        """Multi-GPU parallel step."""
        # This is a simplified version - full implementation would
        # shard the computation across devices

        # For now, just use data parallelism for ensemble runs
        # Real domain decomposition would require more complex sharding

        logger.warning("Full multi-GPU domain decomposition not yet implemented")
        return self._step_single_gpu(state, dt, **kwargs)

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics in MB
        """
        if self.device.platform != "gpu":
            return {"info": "Memory tracking only available for GPU"}

        # Get memory stats from JAX
        stats = {}
        try:
            # This is backend-specific
            backend = jax.default_backend()
            if backend == "gpu":
                # Get CUDA memory info if available
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                stats = {
                    "total_mb": info.total / (1024 * 1024),
                    "used_mb": info.used / (1024 * 1024),
                    "free_mb": info.free / (1024 * 1024),
                    "utilization_percent": 100 * info.used / info.total,
                }
        except ImportError:
            stats = {"info": "pynvml not available for detailed memory stats"}
        except Exception as e:
            stats = {"error": str(e)}

        return stats


def benchmark_gpu_performance(
    grid_sizes: Optional[list[int]] = None, device_type: str = "auto"
) -> dict[str, Any]:
    """Benchmark solver performance on different grid sizes.

    Args:
        grid_sizes: List of grid sizes to test
        device_type: Device to benchmark

    Returns:
        Dictionary with benchmark results
    """
    import time

    from pygsquig.core.grid import make_grid
    from pygsquig.core.solver import gSQGSolver

    # Setup device
    if grid_sizes is None:
        grid_sizes = [128, 256, 512, 1024]
    device = setup_device(device_type)

    results = {
        "device": str(device),
        "platform": device.platform,
        "grid_sizes": grid_sizes,
        "times": {},
        "throughput": {},
    }

    for N in grid_sizes:
        logger.info(f"Benchmarking N={N}")

        # Create solver
        grid = make_grid(N, 2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-8, p=8)
        gpu_solver = GPUOptimizedSolver(solver, device=device)

        # Initialize
        state = solver.initialize(seed=42)

        # Warmup
        for _ in range(3):
            state = gpu_solver.step(state, 0.001)

        # Time multiple steps
        n_steps = 10
        start = time.perf_counter()

        for _ in range(n_steps):
            state = gpu_solver.step(state, 0.001)

        # Ensure computation is complete
        jax.block_until_ready(state["theta_hat"])

        elapsed = time.perf_counter() - start
        time_per_step = elapsed / n_steps

        # Calculate throughput (grid points per second)
        throughput = N * N / time_per_step

        results["times"][N] = time_per_step
        results["throughput"][N] = throughput

        logger.info(f"  Time per step: {time_per_step*1000:.2f} ms")
        logger.info(f"  Throughput: {throughput/1e6:.2f} Mpoints/s")

        # Memory usage
        if device.platform == "gpu":
            mem = gpu_solver.get_memory_usage()
            results[f"memory_N{N}"] = mem

    return results


# Decorator for automatic GPU optimization
def gpu_optimized(func):
    """Decorator to automatically optimize function for GPU execution.

    This ensures the function runs on GPU if available and uses
    optimal memory layout.
    """

    @partial(jax.jit, static_argnums=(1,))
    def wrapper(theta_hat, grid, *args, **kwargs):
        # Ensure complex64 for large grids to save memory
        if grid.N > 1024 and theta_hat.dtype == jnp.complex128:
            theta_hat = theta_hat.astype(jnp.complex64)
            result = func(theta_hat, grid, *args, **kwargs)
            # Handle tuple returns
            if isinstance(result, tuple):
                return tuple(
                    r.astype(jnp.complex128) if hasattr(r, "astype") else r for r in result
                )
            else:
                return result.astype(jnp.complex128) if hasattr(result, "astype") else result
        else:
            return func(theta_hat, grid, *args, **kwargs)

    return wrapper

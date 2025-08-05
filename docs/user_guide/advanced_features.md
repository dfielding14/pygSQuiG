# Advanced Features

This guide covers advanced features in pygSQuiG including GPU acceleration, adaptive timestepping, multi-GPU parallelism, and performance optimization.

## GPU Acceleration

### Automatic GPU Detection

pygSQuiG automatically detects and uses GPUs when available:

```python
from pygsquig.core.gpu_utils import setup_device, get_available_devices

# List available devices
devices = get_available_devices()
print(f"Available devices: {devices}")

# Auto-select best device
device = setup_device("auto")
print(f"Using device: {device}")
```

### GPU-Optimized Solver

```python
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.gpu_utils import GPUOptimizedSolver

# Create base solver
base_solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-8, p=8)

# Wrap with GPU optimization
gpu_solver = GPUOptimizedSolver(
    base_solver,
    device="gpu:0",              # Specific GPU
    memory_optimization=True,     # Enable memory tricks
    use_mixed_precision=False    # Keep full precision
)

# Use exactly like base solver
state = gpu_solver.step(state, dt)
```

### Memory Management

For large simulations:

```python
from pygsquig.core.gpu_utils import optimize_memory_layout

# Get memory optimization parameters
params = optimize_memory_layout(N=2048)
print(f"Recommended settings: {params}")

# Configure JAX
if params['use_float32']:
    jax.config.update('jax_enable_x64', False)

# Set memory allocation
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

### Performance Monitoring

```python
# Monitor GPU usage
mem_stats = gpu_solver.get_memory_usage()
print(f"GPU memory used: {mem_stats['used_gb']:.2f} GB")
print(f"GPU utilization: {mem_stats['utilization']:.1f}%")

# Benchmark performance
from pygsquig.core.gpu_utils import benchmark_gpu_performance

results = benchmark_gpu_performance(
    grid_sizes=[256, 512, 1024, 2048],
    device_type="gpu"
)
```

## Adaptive Timestepping

### Basic Usage

```python
from pygsquig.core.adaptive_solver import AdaptivegSQGSolver
from pygsquig.core.adaptive_timestep import CFLConfig

# Configure CFL parameters
cfl_config = CFLConfig(
    cfl_safety=0.8,      # Safety factor
    target_cfl=0.5,      # Target CFL number
    dt_min=1e-8,         # Minimum timestep
    dt_max=0.01,         # Maximum timestep
    growth_factor=1.1,   # Max growth per step
    shrink_factor=0.5    # Shrink on instability
)

# Create adaptive solver
solver = AdaptivegSQGSolver(
    grid, alpha=1.0, nu_p=1e-8, p=8,
    cfl_config=cfl_config,
    verbose=True
)

# Run with automatic timestep adjustment
state = solver.initialize(seed=42)
results = solver.evolve(
    state,
    t_final=100.0,
    save_interval=1.0
)
```

### Monitoring Timestep Evolution

```python
# Get timestep statistics
stats = solver.timestepper.get_statistics()
print(f"Mean timestep: {stats['dt_mean']:.3e}")
print(f"CFL efficiency: {stats['efficiency']*100:.1f}%")
print(f"Rejected steps: {stats['n_rejected']}")

# Plot timestep history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.semilogy(
    solver.timestepper.time_history,
    solver.timestepper.dt_history
)
plt.xlabel('Time')
plt.ylabel('Timestep')
plt.title('Adaptive Timestep Evolution')
plt.show()
```

### Custom Stability Criteria

```python
from pygsquig.core.adaptive_timestep import AdaptiveTimestepper

class CustomTimestepper(AdaptiveTimestepper):
    def check_stability(self, state_before, state_after, dt_used):
        """Custom stability check."""
        # Default checks
        is_stable, reason = super().check_stability(
            state_before, state_after, dt_used
        )
        
        if not is_stable:
            return is_stable, reason
            
        # Add custom check (e.g., enstrophy growth)
        Z_before = compute_enstrophy(state_before)
        Z_after = compute_enstrophy(state_after)
        
        if Z_after > 10 * Z_before:
            return False, "Excessive enstrophy growth"
            
        return True, None
```

## Multi-GPU Parallelism

### Ensemble Simulations

Run multiple simulations across GPUs:

```python
from pygsquig.core.multi_gpu import MultiGPUSolver, ParallelConfig

# Configure for 4 GPUs
config = ParallelConfig(
    n_devices=4,
    ensemble_size=16,    # 4 runs per GPU
    mode='ensemble'
)

# Create multi-GPU solver
multi_solver = MultiGPUSolver(
    grid=grid,
    alpha=1.0,
    nu_p=1e-8,
    p=8,
    parallel_config=config
)

# Run ensemble with different initial conditions
results = multi_solver.run_ensemble(
    n_steps=10000,
    dt=0.001,
    save_interval=100,
    seed_base=42  # Seeds: 42, 43, 44, ...
)

# Analyze ensemble statistics
from pygsquig.core.multi_gpu import ensemble_statistics

stats = ensemble_statistics(results['states'], grid, alpha)
print(f"Mean energy: {stats['energy_mean']:.3f} ± {stats['energy_std']:.3f}")
```

### Data Parallelism

Process multiple fields in parallel:

```python
# Process batch of states
states_batch = [state1, state2, state3, state4]

results = multi_solver.process_batch(
    states_batch,
    operation='compute_spectrum',
    combine='stack'
)

# Results shape: (4, n_k_bins)
```

### Domain Decomposition (Experimental)

Split large domain across GPUs:

```python
config = ParallelConfig(
    n_devices=4,
    mode='domain_decomposition',
    decomposition='slab',  # or 'pencil'
    halo_size=3
)

# Note: Full implementation requires careful FFT handling
```

## Performance Optimization

### JIT Compilation

pygSQuiG now includes extensive JIT compilation for performance. Key functions are automatically JIT-compiled:

#### Automatically JIT-Compiled Functions

The following functions are JIT-compiled by default for optimal performance:

```python
# Grid operations (all JIT-compiled)
from pygsquig.core.grid import fft2, ifft2, rfft2, irfft2

# These execute on GPU/TPU when available
theta_hat = fft2(theta)  # Fast Fourier transform
theta = ifft2(theta_hat)  # Inverse FFT

# Gradient computations (JIT-compiled)
from pygsquig.core.grid import gradient, divergence, laplacian, curl

grad_x, grad_y = gradient(field_hat, grid)
div = divergence(u_hat, v_hat, grid)
lap = laplacian(field_hat, grid)
vort = curl(u_hat, v_hat, grid)
```

#### Solver Operations

Core solver operations are JIT-compiled:

```python
# Fractional Laplacian (JIT-compiled)
from pygsquig.core.operators import compute_fractional_laplacian
psi_hat = compute_fractional_laplacian(theta_hat, grid, -alpha/2)

# Velocity computation (JIT-compiled)
from pygsquig.core.operators import compute_velocity
u, v = compute_velocity(psi_hat, grid)

# RHS computation (JIT-compiled internally)
solver = gSQGSolver(grid, alpha, nu_p, p)
# step() uses JIT-compiled RHS evaluation
state = solver.step(state, dt)
```

#### Passive Scalar Operations

Scalar advection and diffusion are JIT-compiled:

```python
# These are automatically JIT-compiled
from pygsquig.scalars.passive_scalar import (
    compute_scalar_advection,
    compute_scalar_diffusion,
    compute_passive_scalar_rhs
)

# JIT-compiled scalar variance computation
from pygsquig.scalars.diagnostics import compute_scalar_variance
variance = compute_scalar_variance(scalar_hat)  # Returns JAX array
```

#### Custom JIT Compilation

For custom operations, use JAX's JIT decorator:

```python
import jax

# Compile entire evolution loop
@jax.jit
def evolve_compiled(state, n_steps, dt):
    """JIT-compiled evolution loop."""
    def step_fn(state, _):
        state = solver.step(state, dt)
        return state, None
        
    final_state, _ = jax.lax.scan(step_fn, state, jnp.arange(n_steps))
    return final_state

# First call compiles (slow)
state_final = evolve_compiled(state, 1000, 0.001)

# Subsequent calls are fast
state_final = evolve_compiled(state, 1000, 0.001)
```

#### JIT Compilation Best Practices

1. **Avoid Python control flow inside JIT functions**:
   ```python
   # Bad - Python if statement
   @jax.jit
   def bad_function(x):
       if x > 0:  # This will fail
           return x
       return -x
   
   # Good - Use JAX control flow
   @jax.jit
   def good_function(x):
       return jnp.where(x > 0, x, -x)
   ```

2. **Use static arguments for shapes**:
   ```python
   @partial(jax.jit, static_argnums=(1, 2))
   def compute_spectrum(field, N, L):
       # N and L are treated as compile-time constants
       k_bins = create_bins(N, L)
       return radial_average(field, k_bins)
   ```

3. **Monitor compilation time**:
   ```python
   import time
   
   # Time the first (compilation) call
   start = time.time()
   result = jit_function(args)
   print(f"Compilation + execution: {time.time() - start:.2f}s")
   
   # Time subsequent calls
   start = time.time()
   result = jit_function(args)
   print(f"Execution only: {time.time() - start:.4f}s")
   ```

### Vectorization

Process multiple operations efficiently:

```python
# Vectorize over parameters
@jax.vmap
def compute_with_params(nu_p):
    """Compute for different viscosities."""
    solver = gSQGSolver(grid, alpha=1.0, nu_p=nu_p, p=8)
    state = initial_state
    for _ in range(100):
        state = solver.step(state, dt)
    return solver.get_diagnostics(state)

# Run for multiple viscosities
nu_p_values = jnp.logspace(-8, -6, 10)
diagnostics = compute_with_params(nu_p_values)
```

### Memory-Efficient Operations

For large grids:

```python
# Chunked operations
def chunked_operation(field, chunk_size=512):
    """Process large field in chunks."""
    N = field.shape[0]
    result = jnp.zeros_like(field)
    
    for i in range(0, N, chunk_size):
        for j in range(0, N, chunk_size):
            chunk = field[i:i+chunk_size, j:j+chunk_size]
            # Process chunk
            result = result.at[i:i+chunk_size, j:j+chunk_size].set(
                process_chunk(chunk)
            )
    return result

# Checkpointing for memory
from jax import checkpoint

@checkpoint  # Trade compute for memory
def memory_intensive_operation(state):
    # Complex operations
    return result
```

## Custom Operators

### Fractional Laplacian Variants

```python
from pygsquig.core.operators import compute_fractional_laplacian

class ModifiedSQGSolver(gSQGSolver):
    def compute_velocity_custom(self, theta_hat, grid):
        """Custom velocity computation."""
        # Standard fractional Laplacian
        psi_hat = -compute_fractional_laplacian(
            theta_hat, grid, -self.alpha/2
        )
        
        # Add modification (e.g., rotation)
        beta = 0.1  # Rotation parameter
        psi_hat = psi_hat + 1j * beta * theta_hat
        
        # Compute velocity
        u = -1j * grid.ky * psi_hat
        v = 1j * grid.kx * psi_hat
        
        return grid.ifft2(u), grid.ifft2(v)
```

### Custom Diagnostics

```python
def compute_palinstrophy(state, grid, alpha):
    """Compute palinstrophy (gradient of enstrophy)."""
    theta_hat = state['theta_hat']
    
    # Compute (-Δ)^(α/2) θ
    lap_theta = compute_fractional_laplacian(theta_hat, grid, alpha/2)
    
    # Compute gradient
    grad_x = grid.ifft2(1j * grid.kx * lap_theta)
    grad_y = grid.ifft2(1j * grid.ky * lap_theta)
    
    # Palinstrophy
    P = jnp.mean(grad_x**2 + grad_y**2)
    return P.real

# Add to solver
solver.add_diagnostic('palinstrophy', compute_palinstrophy)
```

## Profiling and Debugging

### Performance Profiling

```python
from pygsquig.scripts.profile_performance import profile_solver

# Profile solver performance
profile_results = profile_solver(
    solver,
    n_steps=100,
    profile_memory=True,
    profile_kernels=True
)

# Show bottlenecks
print("Slowest operations:")
for op, time in profile_results['timings'].items():
    print(f"  {op}: {time:.3f}s")
```

### JAX Debugging

```python
# Enable NaN checking
from jax import config
config.update("jax_debug_nans", True)

# Disable JIT for debugging
with jax.disable_jit():
    state = solver.step(state, dt)

# Print intermediate values
def debug_step(state, dt):
    theta_hat = state['theta_hat']
    print(f"Max |θ̂|: {jnp.max(jnp.abs(theta_hat))}")
    
    # Check for issues
    if jnp.any(jnp.isnan(theta_hat)):
        raise ValueError("NaN detected!")
        
    return solver.step(state, dt)
```

## Integration with External Tools

### Export to xarray

```python
import xarray as xr

def state_to_xarray(state, grid):
    """Convert state to xarray Dataset."""
    theta = grid.ifft2(state['theta_hat']).real
    
    ds = xr.Dataset(
        {
            'theta': (['x', 'y'], theta),
            'time': state['time'],
            'energy': compute_total_energy(state['theta_hat'], grid)
        },
        coords={
            'x': grid.x[:, 0],
            'y': grid.y[0, :],
        },
        attrs={
            'alpha': solver.alpha,
            'nu_p': solver.nu_p,
            'step': state['step']
        }
    )
    return ds
```

### Parallel I/O

```python
from pygsquig.io.parallel_io import ParallelHDF5Writer

# For large parallel runs
writer = ParallelHDF5Writer(
    filename="large_simulation.h5",
    n_ranks=4,
    rank=rank  # MPI rank or GPU ID
)

# Each rank writes its portion
writer.write_chunk(data_chunk, offset, size)
writer.close()
```

## Best Practices

1. **GPU Memory**: Monitor usage, use float32 for large grids
2. **Compilation**: JIT compile hot loops
3. **Vectorization**: Batch similar operations
4. **Checkpointing**: Save state for restart
5. **Profiling**: Profile before optimizing
6. **Scaling**: Test weak and strong scaling
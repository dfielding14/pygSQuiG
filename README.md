# pygSQuiG

A Python/JAX implementation of a spectral solver for the generalized Surface Quasi-Geostrophic (gSQG) equations, designed for high-performance turbulence simulations on GPUs.

## Overview

pygSQuiG solves the generalized Surface Quasi-Geostrophic equations:

```
∂_t θ + u·∇θ = F - D
```

where the velocity field is related to the scalar field θ through:

```
u = ∇^⊥(-Δ)^(-α/2)θ
```

The parameter α ∈ [-2, 2] controls the relationship between θ and velocity:
- α = 0: 2D Euler equations
- α = 1: Surface Quasi-Geostrophic (SQG) equations
- α = 2: Modified SQG with logarithmic kernel

## Features

- **High Performance**: Built on JAX for automatic differentiation and JIT compilation
- **GPU Acceleration**: Seamlessly runs on GPUs for large-scale simulations
- **Optimized Solver**: Includes performance-optimized version with batch processing
- **Spectral Accuracy**: Pseudo-spectral methods with 2/3 dealiasing
- **Flexible Physics**: Supports the full gSQG family (-2 ≤ α ≤ 2)
- **Modern Design**: Type-safe, functional programming approach
- **Comprehensive I/O**: HDF5/xarray integration for data management

## Installation

### Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.0
- NumPy
- h5py
- xarray

### Install from source

```bash
git clone https://github.com/yourusername/pygSQuiG.git
cd pygSQuiG
pip install -e .
```

## Quick Start

### Basic Simulation

```python
import jax.numpy as jnp
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver

# Create grid
N = 256  # Resolution
L = 2 * jnp.pi  # Domain size
grid = make_grid(N, L)

# Initialize solver for SQG (α = 1)
solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-4, p=8)

# Set initial condition
theta0 = jnp.sin(4 * grid.x) * jnp.cos(4 * grid.y)
state = solver.initialize(theta0)

# Run simulation
dt = 0.001
for _ in range(1000):
    state = solver.step(state, dt)
```

### Using the CLI

Run a simulation from a YAML configuration file:

```bash
pygsquig-run config.yml --device gpu --output results/
```

Example configuration (`config.yml`):

```yaml
grid:
  N: 512
  L: 6.283185307179586  # 2π

solver:
  alpha: 1.0  # SQG
  nu_p: 1.0e-8
  p: 8

forcing:
  type: ring
  kf: 40.0
  dk: 2.0
  epsilon: 0.1

simulation:
  dt: 0.001
  t_end: 100.0
  
output:
  save_every: 1.0
  fields: ['theta', 'energy_spectrum']
```

## Advanced Usage

### Forced-Dissipative Turbulence

```python
from pygsquig.forcing.ring_forcing import RingForcing
from pygsquig.forcing.damping import CombinedDamping

# Create forcing at wavenumber kf = 40
forcing = RingForcing(kf=40.0, dk=2.0, epsilon=0.1)

# Add large-scale damping and hyperviscosity
damping = CombinedDamping(mu=0.1, kf=40.0, nu_p=1e-8, p=8)

# Time step with forcing and damping
import jax
key = jax.random.PRNGKey(42)
state = solver.step(state, dt, forcing=forcing, damping=damping, key=key, grid=grid)
```

### Computing Diagnostics

```python
from pygsquig.utils.diagnostics import (
    compute_energy_spectrum,
    compute_total_energy,
    compute_enstrophy
)

# Energy spectrum
k_bins, E_k = compute_energy_spectrum(state['theta_hat'], grid, alpha=1.0)

# Total energy
energy = compute_total_energy(state['theta_hat'], grid, alpha=1.0)

# Enstrophy
enstrophy = compute_enstrophy(state['theta_hat'], grid, alpha=1.0)
```

### Checkpointing and Restart

```python
from pygsquig.io.hdf5_io import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint('checkpoint.h5', state, solver_params={'alpha': 1.0})

# Restart from checkpoint
state, solver_params = load_checkpoint('checkpoint.h5')
```

## Project Structure

```
pygSQuiG/
├── pygsquig/
│   ├── core/           # Core numerical methods
│   │   ├── grid.py     # Grid and FFT operations
│   │   ├── operators.py # Spectral operators
│   │   ├── solver.py   # Main solver class
│   │   └── time_integrator.py # Time stepping schemes
│   ├── forcing/        # Forcing and dissipation
│   │   ├── ring_forcing.py # Ring forcing in Fourier space
│   │   └── damping.py  # Large-scale and hyperviscous damping
│   ├── io/            # Input/output
│   │   ├── config.py   # Configuration management
│   │   └── hdf5_io.py  # HDF5/xarray I/O
│   ├── utils/         # Utilities
│   │   └── diagnostics.py # Diagnostic computations
│   └── scripts/       # Command-line tools
│       ├── run.py     # Main simulation runner
│       └── validate.py # Validation tests
├── tests/             # Comprehensive test suite
├── examples/          # Example configurations
└── docs/             # Documentation
```

## Performance Tips

1. **Use GPU acceleration**: Set `JAX_PLATFORM_NAME=gpu` or use `--device gpu`
2. **Enable double precision**: The solver uses float64 by default for accuracy
3. **Batch operations**: Process multiple realizations using `jax.vmap`
4. **Monitor CFL**: Use adaptive timestepping to maintain stability

## Physics Background

The gSQG equations describe the evolution of an active scalar θ (e.g., temperature, vorticity) that is advected by a velocity field derived from θ itself. Key features:

- **Energy cascade**: Energy flows from large to small scales
- **Conservation laws**: Energy, enstrophy (in inviscid limit)
- **Spectral slopes**: E(k) ~ k^(-5/3) for SQG turbulence

## Performance Optimization

pygSQuiG includes an optimized solver that provides significant performance improvements:

### Using the Optimized Solver

```python
from pygsquig.core.solver_optimized import gSQGSolverOptimized

# Create optimized solver (same interface as regular solver)
solver = gSQGSolverOptimized(grid, alpha=1.0, nu_p=1e-4, p=8)
state = solver.initialize(seed=42)

# Single stepping (1.3-1.4x faster)
state = solver.step(state, dt)

# Multi-step integration (up to 10x faster for large batches)
state = solver.multistep(state, n_steps=1000, dt=0.001)
```

### Performance Tips

1. **Use multistep integration** for simulations without forcing/damping
2. **Batch operations** - process many steps at once
3. **GPU acceleration** - Set `JAX_PLATFORM_NAME=gpu` for larger grids
4. **Profile your code** - Use `pygsquig.scripts.profile_performance`

### Benchmarks

Performance on CPU (Intel i7):
- 256×256 grid: ~156 steps/sec (optimized) vs ~111 steps/sec (original)
- Multistep with batch=1000: ~1640 steps/sec
- GPU provides 5-10x additional speedup for N≥512

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Testing

Run the test suite:

```bash
pytest tests/
```

Run validation tests:

```bash
python -m pygsquig.scripts.validate
```

## Citation

If you use pygSQuiG in your research, please cite:

```bibtex
@software{pygsquig,
  title = {pygSQuiG: A JAX-based spectral solver for generalized SQG turbulence},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pygSQuiG}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project was developed using modern software engineering practices with AI assistance from Claude (Anthropic).
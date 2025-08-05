# pygSQuiG User Guide

Welcome to the pygSQuiG (Python generalized Surface Quasi-Geostrophic) solver documentation. This guide will help you get started with simulating geophysical turbulence using our JAX-based spectral solver.

## Table of Contents

1. [Getting Started](getting_started.md) - Installation and first simulation
2. [Core Concepts](core_concepts.md) - Understanding gSQG equations and numerics
3. [Configuration](configuration.md) - Setting up simulations
4. [Running Simulations](running_simulations.md) - Execution and monitoring
5. [Forcing Patterns](forcing_patterns.md) - Energy injection methods
6. [Passive Scalars](passive_scalars_guide.md) - Tracer dynamics
7. [Advanced Features](advanced_features.md) - GPU, adaptive timestepping
8. [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Quick Example

Here's a minimal example to run your first gSQG simulation:

```python
import numpy as np
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing.ring_forcing import RingForcing

# Create grid
N = 256  # Resolution
L = 2 * np.pi  # Domain size
grid = make_grid(N, L)

# Create solver
solver = gSQGSolver(
    grid=grid,
    alpha=1.0,      # SQG turbulence
    nu_p=1e-6,      # Hyperviscosity
    p=8             # Hyperviscosity order
)

# Initialize with random field
state = solver.initialize(seed=42)

# Add forcing
forcing = RingForcing(kf=30.0, dk=2.0, epsilon=0.1)

# Run simulation
dt = 0.001
for step in range(1000):
    state = solver.step(state, dt, forcing=forcing)
    if step % 100 == 0:
        print(f"Step {step}, time = {state['time']:.2f}")
```

## Features

### Physical Models
- **Generalized SQG**: Fractional Laplacian with adjustable α ∈ [0, 2]
- **Forcing**: Deterministic and stochastic patterns
- **Dissipation**: Hyperviscosity and large-scale damping
- **Passive scalars**: Multiple tracer advection with sources and reactions

### Numerical Methods
- **Pseudo-spectral**: Fourier space operations with JIT compilation
- **Time integration**: RK4, SSP-RK3, adaptive timestepping
- **Dealiasing**: 2/3 rule for nonlinear terms
- **Conservation**: Energy and enstrophy diagnostics

### Performance
- **JAX backend**: Automatic differentiation and extensive JIT compilation
- **GPU support**: CUDA/ROCm acceleration for large simulations
- **Optimized operations**: Key functions are pre-compiled for speed
- **Memory efficiency**: Careful memory management for large grids

### Analysis
- **Diagnostics**: Energy spectra, fluxes, structure functions
- **Visualization**: Field plots, animations, spectral analysis
- **I/O**: HDF5 format with xarray compatibility

## Installation

### Using pip
```bash
pip install pygsquig
```

### Using Docker
```bash
# CPU version
docker pull pygsquig:latest

# GPU version
docker pull pygsquig-gpu:latest
```

### From source
```bash
git clone https://github.com/yourusername/pygSQuiG.git
cd pygSQuiG
pip install -e .
```

## Getting Help

- **Documentation**: This guide and API reference
- **Examples**: See the `examples/` directory
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions on GitHub

## Citation

If you use pygSQuiG in your research, please cite:

```bibtex
@software{pygsquig2024,
  title = {pygSQuiG: Python generalized Surface Quasi-Geostrophic solver},
  author = {Your Name and Contributors},
  year = {2024},
  url = {https://github.com/yourusername/pygSQuiG}
}
```

## License

pygSQuiG is released under the MIT License. See [LICENSE](../../LICENSE) for details.
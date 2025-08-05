# Getting Started with pygSQuiG

This guide will walk you through installing pygSQuiG and running your first simulation.

## Installation

### Requirements

- Python 3.9 or higher
- JAX (automatically installed)
- NumPy, SciPy, h5py
- Optional: CUDA toolkit for GPU support

### Install Methods

#### 1. Using pip (Recommended)

```bash
# CPU-only version
pip install pygsquig

# GPU version (requires CUDA)
pip install pygsquig[gpu]
```

#### 2. Using Docker

```bash
# Pull the image
docker pull pygsquig:latest

# Run with Jupyter
docker run -it -p 8888:8888 pygsquig:latest
```

#### 3. From Source

```bash
# Clone repository
git clone https://github.com/yourusername/pygSQuiG.git
cd pygSQuiG

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with GPU support
pip install -e .[gpu]
```

### Verify Installation

```python
import pygsquig
print(f"pygSQuiG version: {pygsquig.__version__}")

# Check JAX backend
import jax
print(f"JAX devices: {jax.devices()}")
```

## Your First Simulation

Let's simulate decaying 2D turbulence:

### 1. Basic Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.diagnostics import compute_energy_spectrum

# Simulation parameters
N = 256                    # Grid resolution
L = 2 * np.pi             # Domain size [0, L] x [0, L]
alpha = 1.0               # SQG (alpha=1), Euler (alpha=0)
nu_p = 1e-8               # Hyperviscosity coefficient
p = 8                     # Hyperviscosity order

# Create grid
grid = make_grid(N, L)

# Create solver
solver = gSQGSolver(grid, alpha, nu_p, p)
```

### 2. Initial Condition

```python
# Initialize with random field
state = solver.initialize(
    init_type='random',
    energy=1.0,           # Target energy
    peak_k=10,            # Peak wavenumber
    seed=42               # For reproducibility
)

# Or load from file
# state = solver.initialize(init_type='file', filename='initial.h5')

# Visualize initial condition
theta_init = grid.ifft2(state['theta_hat']).real
plt.figure(figsize=(8, 6))
plt.imshow(theta_init, cmap='RdBu_r', extent=[0, L, 0, L])
plt.colorbar(label='θ')
plt.title('Initial Condition')
plt.show()
```

### 3. Time Integration

```python
# Time stepping parameters
dt = 0.001               # Time step
n_steps = 1000          # Number of steps
save_interval = 100     # Save every N steps

# Storage for diagnostics
times = []
energies = []
enstrophies = []

# Main time loop
for step in range(n_steps):
    # Advance one time step
    state = solver.step(state, dt)
    
    # Save diagnostics
    if step % save_interval == 0:
        t = state['time']
        diagnostics = solver.get_diagnostics(state)
        
        times.append(t)
        energies.append(diagnostics['energy'])
        enstrophies.append(diagnostics['enstrophy'])
        
        print(f"Step {step:4d}, t = {t:6.2f}, "
              f"E = {diagnostics['energy']:.4f}, "
              f"Z = {diagnostics['enstrophy']:.4f}")
```

### 4. Analysis

```python
# Plot energy evolution
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(times, energies, 'b-', label='Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Evolution')
plt.grid(True)

plt.subplot(122)
plt.plot(times, enstrophies, 'r-', label='Enstrophy')
plt.xlabel('Time')
plt.ylabel('Enstrophy')
plt.title('Enstrophy Evolution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Energy spectrum
k_bins, E_k = compute_energy_spectrum(state['theta_hat'], grid)

plt.figure(figsize=(8, 6))
plt.loglog(k_bins, E_k, 'b-', linewidth=2)
plt.loglog(k_bins, k_bins**(-5/3), 'k--', label='k^{-5/3}')
plt.loglog(k_bins, k_bins**(-3), 'k:', label='k^{-3}')
plt.xlabel('Wavenumber k')
plt.ylabel('Energy E(k)')
plt.title('Energy Spectrum')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Forced Turbulence Example

For statistically stationary turbulence:

```python
from pygsquig.forcing.ring_forcing import RingForcing
import jax

# Create forcing
forcing = RingForcing(
    kf=30.0,              # Forcing wavenumber
    dk=2.0,               # Bandwidth
    epsilon=0.1           # Energy injection rate
)

# Time stepping with forcing
key = jax.random.PRNGKey(123)

for step in range(10000):
    # Split random key
    key, subkey = jax.random.split(key)
    
    # Step with forcing
    state = solver.step(state, dt, forcing=forcing, key=subkey)
    
    if step % 1000 == 0:
        print(f"Step {step}, t = {state['time']:.1f}")
```

## Saving Results

```python
from pygsquig.io.hdf5_io import HDF5Writer

# Create writer
writer = HDF5Writer(
    filename="simulation_data.h5",
    grid=grid,
    compression='gzip'
)

# Save metadata
metadata = {
    'alpha': alpha,
    'nu_p': nu_p,
    'p': p,
    'dt': dt
}
writer.write_metadata(metadata)

# Save state
writer.write_state(state, diagnostics)

# Close file
writer.close()
```

## Next Steps

Now that you've run your first simulation:

1. Learn about [Core Concepts](core_concepts.md) - Understand the physics and numerics
2. Explore [Configuration](configuration.md) - Set up complex simulations
3. Try different [Forcing Patterns](forcing_patterns.md) - Drive turbulence
4. Add [Passive Scalars](passive_scalars_guide.md) - Study mixing
5. Use [Advanced Features](advanced_features.md) - GPU acceleration, adaptive timestepping

## Common Issues

### Import Error
```python
# If you get: ModuleNotFoundError: No module named 'pygsquig'
# Make sure you're in the right environment:
which python  # Should show your venv path
```

### Memory Error
```python
# For large simulations, use float32:
import jax
jax.config.update('jax_enable_x64', False)
```

### GPU Not Found
```python
# Check GPU availability:
import jax
print(jax.devices())  # Should show GPU devices

# Force CPU if needed:
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

## Example Notebooks

Check the `notebooks/` directory for interactive examples:
- `sqg_tutorial.ipynb` - Step-by-step tutorial
- `forced_turbulence.ipynb` - Energy cascade studies
- `passive_scalars.ipynb` - Mixing and transport
- `gpu_optimization.ipynb` - Performance tuning
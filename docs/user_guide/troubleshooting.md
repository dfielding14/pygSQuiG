# Troubleshooting Guide

This guide helps diagnose and fix common issues when using pygSQuiG.

## Installation Issues

### ImportError: No module named 'pygsquig'

**Problem**: Python can't find pygSQuiG after installation.

**Solutions**:
```bash
# Check if installed
pip list | grep pygsquig

# Ensure you're in the right environment
which python
python -c "import sys; print(sys.path)"

# Reinstall in development mode
cd /path/to/pygSQuiG
pip install -e .
```

### JAX Installation Problems

**Problem**: JAX not installing or GPU support not working.

**Solutions**:
```bash
# For CPU-only
pip install --upgrade jax

# For GPU (CUDA 12)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Check JAX installation
python -c "import jax; print(jax.devices())"
```

### Version Conflicts

**Problem**: Dependency version conflicts.

**Solution**: Use a fresh virtual environment:
```bash
python -m venv pygsquig_env
source pygsquig_env/bin/activate  # Windows: pygsquig_env\Scripts\activate
pip install --upgrade pip
pip install pygsquig
```

## Runtime Errors

### Out of Memory (OOM)

**Symptoms**: 
- `RuntimeError: Resource exhausted: Out of memory`
- System freeze or crash

**Solutions**:

1. **Reduce grid resolution**:
```python
# Start with smaller grid
N = 256  # Instead of 512 or 1024
```

2. **Use float32 precision**:
```python
import jax
jax.config.update('jax_enable_x64', False)
```

3. **Limit GPU memory**:
```python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
```

4. **Chunk operations**:
```python
from pygsquig.core.gpu_utils import optimize_memory_layout
params = optimize_memory_layout(N)
```

### NaN or Inf Values

**Symptoms**:
- `RuntimeError: Invalid value (nan/inf) in array`
- Simulation blows up

**Causes and Solutions**:

1. **Timestep too large**:
```python
# Reduce timestep
dt = 0.0001  # Try 10x smaller

# Or use adaptive timestepping
from pygsquig.core.adaptive_solver import AdaptivegSQGSolver
solver = AdaptivegSQGSolver(grid, alpha, nu_p, p)
```

2. **Insufficient dissipation**:
```python
# Increase hyperviscosity
nu_p = 1e-6  # Instead of 1e-8

# Or increase order
p = 16  # Instead of 8
```

3. **Initial condition too strong**:
```python
# Reduce initial energy
state = solver.initialize(energy=0.1)  # Instead of 1.0
```

4. **Debug mode**:
```python
# Enable NaN checking
from jax import config
config.update("jax_debug_nans", True)

# Add diagnostics
def debug_step(solver, state, dt):
    print(f"Before: max|θ| = {np.max(np.abs(state['theta_hat'])):.3e}")
    new_state = solver.step(state, dt)
    print(f"After: max|θ| = {np.max(np.abs(new_state['theta_hat'])):.3e}")
    return new_state
```

### Slow Performance

**Symptoms**: Simulation runs much slower than expected.

**Diagnostics**:
```python
import time

# Time single step
start = time.time()
state = solver.step(state, dt)
print(f"Step time: {time.time() - start:.3f}s")

# Check if using GPU
import jax
print(f"Devices: {jax.devices()}")
```

**Solutions**:

1. **Ensure JIT compilation**:
```python
# First step is slow (compilation), rest should be fast
for i in range(10):
    t0 = time.time()
    state = solver.step(state, dt)
    print(f"Step {i}: {time.time() - t0:.3f}s")
```

2. **Check device placement**:
```python
# Force GPU usage
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

# Or force CPU if GPU is problematic
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

3. **Profile performance**:
```python
from pygsquig.scripts.profile_performance import profile_solver
profile_results = profile_solver(solver, n_steps=10)
```

## Numerical Issues

### Energy Not Conserved

**Problem**: Energy grows or decays unexpectedly.

**Check conservation**:
```python
# Monitor energy
energies = []
for step in range(1000):
    state = solver.step(state, dt)
    E = compute_total_energy(state['theta_hat'], grid)
    energies.append(E)
    
    if step % 100 == 0:
        drift = (E - energies[0]) / energies[0]
        print(f"Energy drift: {drift*100:.2f}%")
```

**Solutions**:
- Reduce timestep
- Check dealiasing is enabled
- Verify forcing/dissipation balance

### Wrong Cascade Direction

**Problem**: Energy flows in unexpected direction.

**Diagnose**:
```python
# Check energy flux
k_bins, flux = compute_energy_flux(state['theta_hat'], grid, alpha)

import matplotlib.pyplot as plt
plt.plot(k_bins, flux)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('k')
plt.ylabel('Energy flux')
plt.show()
```

**Solutions**:
- Verify α parameter (α < 2/3 → inverse cascade)
- Check forcing scale
- Ensure adequate resolution

### Spectrum Shape Wrong

**Problem**: Energy spectrum doesn't match expected power law.

**Check spectrum**:
```python
k_bins, E_k = compute_energy_spectrum(state['theta_hat'], grid)

plt.loglog(k_bins, E_k, 'b-', label='Computed')
plt.loglog(k_bins, k_bins**(-5/3), 'k--', label='k^{-5/3}')
plt.loglog(k_bins, k_bins**(-3), 'k:', label='k^{-3}')
plt.legend()
plt.show()
```

**Solutions**:
- Run longer (may not be equilibrated)
- Adjust forcing parameters
- Check Reynolds number

## Configuration Problems

### YAML Parse Errors

**Problem**: `yaml.scanner.ScannerError`

**Common causes**:
- Incorrect indentation (use spaces, not tabs)
- Missing colons after keys
- Unquoted special characters

**Validate YAML**:
```python
import yaml

# Test parse
with open('config.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
        print("YAML valid!")
    except yaml.YAMLError as e:
        print(f"Error: {e}")
```

### Invalid Parameter Combinations

**Problem**: `ConfigurationError: Invalid configuration`

**Common issues**:
```python
# Resolution must be even
N = 256  # Good
N = 255  # Bad

# CFL safety must be < 1
cfl_safety = 0.8  # Good
cfl_safety = 1.2  # Bad

# Wavenumbers must be resolved
kf = 40    # Good for N=256
kf = 200   # Bad - above Nyquist
```

## I/O Issues

### HDF5 Write Errors

**Problem**: Can't write output files.

**Solutions**:

1. **Check permissions**:
```bash
# Check directory exists and is writable
ls -la ./data
mkdir -p ./data
chmod 755 ./data
```

2. **Close files properly**:
```python
writer = HDF5Writer("output.h5")
try:
    writer.write_state(state)
finally:
    writer.close()  # Always close!
```

3. **Handle existing files**:
```python
# Append mode
writer = HDF5Writer("output.h5", mode='a')

# Or remove old file
import os
if os.path.exists("output.h5"):
    os.remove("output.h5")
```

### Large File Sizes

**Problem**: Output files are too large.

**Solutions**:

1. **Enable compression**:
```python
writer = HDF5Writer(
    "output.h5",
    compression='gzip',
    compression_level=6
)
```

2. **Save selectively**:
```yaml
output:
  save_fields: false      # Don't save full fields
  save_spectra: true      # Only spectra
  save_interval: 10.0     # Less frequent
```

3. **Use chunking**:
```python
writer = HDF5Writer(
    "output.h5",
    chunk_size=(128, 128),
    chunk_cache_size=32*1024*1024  # 32MB cache
)
```

## Docker Issues

### Container Won't Start

**Problem**: Docker container fails to run.

**Debug**:
```bash
# Check Docker is running
docker ps

# Check image exists
docker images | grep pygsquig

# Run with debug output
docker run --rm -it pygsquig:latest /bin/bash
```

### GPU Not Available in Container

**Problem**: Container can't access GPU.

**Solutions**:
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi

# Use GPU-enabled image
./docker/build.sh --gpu
./docker/run.sh --gpu
```

## Getting Help

### Debug Information

When reporting issues, include:

```python
# System info
import platform
print(f"Python: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

# Package versions
import pygsquig
import jax
import numpy as np
print(f"pygSQuiG: {pygsquig.__version__}")
print(f"JAX: {jax.__version__}")
print(f"NumPy: {np.__version__}")

# JAX configuration
print(f"Devices: {jax.devices()}")
print(f"64-bit: {jax.config.jax_enable_x64}")

# Minimal reproducing example
# Include smallest code that shows the problem
```

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `Resource exhausted` | Out of memory | Reduce grid size or use float32 |
| `Invalid argument` | Wrong parameter type | Check parameter types match expected |
| `NaN detected` | Numerical instability | Reduce timestep or increase dissipation |
| `CUDA error` | GPU problem | Update drivers or use CPU |
| `Incompatible shapes` | Array mismatch | Check grid dimensions are consistent |

### Community Support

- GitHub Issues: Report bugs with minimal examples
- Discussions: Ask questions and share solutions
- Documentation: Check latest docs for updates

Remember: Most issues have been encountered before - search existing issues first!
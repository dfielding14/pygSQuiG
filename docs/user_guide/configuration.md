# Configuration Guide

This guide covers how to configure pygSQuiG simulations using YAML files or Python dictionaries.

## Configuration Structure

pygSQuiG uses a hierarchical configuration with these main sections:

```yaml
grid:            # Grid parameters
solver:          # Solver configuration
forcing:         # Forcing configuration  
scalars:         # Passive scalars (optional)
output:          # Output and I/O settings
simulation:      # Simulation control
initial_condition: # Initial conditions
```

## Grid Configuration

### Grid Parameters

```yaml
grid:
  N: 512                        # Number of grid points (must be even)
  L: 6.283185307179586         # Domain size (2π)
```

## Solver Configuration

### Basic Solver Parameters

```yaml
solver:
  alpha: 1.0                    # Fractional Laplacian exponent (0-2)
                               # 0.0 = 2D Euler, 1.0 = SQG, 2.0 = passive scalar
  
  dissipation:
    type: hyperviscosity       # Type of dissipation
    nu_p: 1.0e-8              # Hyperviscosity coefficient
    p: 8                       # Hyperviscosity order (∇^p dissipation)
  
  damping:                     # Optional large-scale damping
    type: linear_drag         # or "none"
    mu: 0.1                   # Linear drag coefficient
    k_cutoff_factor: 0.5      # Apply for k < kf * factor
  
  time_integration:
    method: RK4               # RK4 or SSP-RK3
    adaptive_cfl: true        # Use adaptive timestepping
    cfl_safety: 0.8          # Safety factor (<1)
    dt: 0.001                # Fixed timestep (if adaptive_cfl: false)
    dt_max: 0.01             # Maximum timestep (optional)
```

## Initial Conditions

```yaml
initial_condition:
  type: random          # Options: random, checkpoint, function
  seed: 42             # Random seed for reproducibility
  amplitude: 1.0       # Amplitude for random IC
  
  # For checkpoint restart
  checkpoint_path: "checkpoint_001000.h5"  # When type: checkpoint
```

## Forcing Configuration

### Ring Forcing (Most Common)

```yaml
forcing:
  type: ring
  kf: 40.0              # Forcing wavenumber
  dk: 2.0               # Bandwidth
  epsilon: 0.1          # Energy injection rate
  tau_f: 0.0           # Correlation time (0 = white noise)
  seed: 42             # Random seed (optional)
```

### Deterministic Forcing

```yaml
forcing:
  type: "deterministic"
  pattern: "taylor_green"  # or kolmogorov, checkerboard, etc.
  amplitude: 1.0
  k: 4                    # Pattern wavenumber
  time_dependence: "steady"  # or oscillatory, growing
  omega: 2.0              # For oscillatory
```

### Stochastic Forcing

```yaml
forcing:
  type: "stochastic"
  pattern: "white_noise"   # or colored_noise, ou_process
  amplitude: 0.5
  k_min: 20.0
  k_max: 40.0
  
  # For colored noise
  spectral_slope: -2.0
  k_peak: 30.0
  
  # For OU process
  correlation_time: 1.0
```

### Physical Forcing

```yaml
forcing:
  type: "physical"
  pattern: "shear_layer"   # or jet, convective_plumes
  
  # For shear layers
  amplitude: 1.0
  shear_width: 0.1
  n_layers: 2
  orientation: "horizontal"
  
  # For jets
  jet_width: 0.15
  n_jets: 3
  meander_amplitude: 0.1
```

### Combined Forcing

```yaml
forcing:
  type: "combined"
  forcings:
    - type: "ring"
      kf: 40.0
      epsilon: 0.1
      weight: 1.0
      
    - type: "physical"
      pattern: "jet"
      amplitude: 0.5
      weight: 0.5
```

## Simulation Control

```yaml
simulation:
  t_end: 100.0                  # End time of simulation
  output_interval: 1.0          # Time between outputs
  checkpoint_interval: 10.0     # Time between checkpoints
  wall_time_limit: 3600        # Max wall time in seconds (optional)
  log_interval: 0.1            # Time between log messages
```

## Output Configuration

```yaml
output:
  fields:                       # Fields to save
    - theta                     # Active scalar (buoyancy)
    - velocity                  # u and v components
    - scalars                   # All passive scalars
    - streamfunction           # Stream function (optional)
    
  diagnostics:                  # Diagnostics to compute
    - energy_spectrum          # Energy spectrum E(k)
    - scalar_flux              # Turbulent scalar flux
    - enstrophy                # Total enstrophy
    - energy_flux              # Energy transfer
    
  save_every_n_steps: null     # Save every N steps (optional)
  compress: true               # Use compression
```

## Diagnostics

```yaml
diagnostics:
  # Console output
  print_interval: 10.0   # Print frequency
  verbosity: "info"      # debug, info, warning, error
  
  # Quantities to track
  track:
    - "energy"          # Total energy
    - "enstrophy"       # Total enstrophy
    - "dissipation"     # Energy dissipation rate
    - "injection"       # Energy injection rate
    - "max_velocity"    # Maximum velocity
    - "cfl_number"      # Current CFL
    
  # Spectral diagnostics
  spectral:
    compute_flux: true  # Energy flux
    compute_transfers: true  # Spectral transfers
    k_shells: 50        # Number of shells
```

## Performance Options

```yaml
performance:
  # Device selection
  device: "auto"        # auto, cpu, gpu, gpu:0
  
  # Memory optimization
  use_float32: false    # Use single precision
  chunk_operations: true # Chunk large operations
  
  # JAX options
  jax_enable_x64: true  # 64-bit precision
  jax_platform_name: null  # Override platform
  
  # GPU-specific
  gpu_memory_fraction: 0.9  # Fraction to use
  enable_cudnn: true     # Use cuDNN
```

## Passive Scalars

```yaml
scalars:
  enabled: true
  species:
    - name: "temperature"
      kappa: 0.01               # Scalar diffusivity
      source:                   # Optional source term
        type: "localized"
        parameters:
          x0: 3.14159
          y0: 3.14159
          sigma: 0.5
          amplitude: 1.0
      initial_condition: "gaussian"
      initial_params:
        center: [3.14159, 3.14159]
        width: 1.0
        
    - name: "dye"
      kappa: 0.001
      source:
        type: "exponential"
        parameters:
          rate: -0.1            # Decay rate
      initial_condition: "random"
      initial_params:
        seed: 123
        amplitude: 0.1
```

See the [Passive Scalars Guide](passive_scalars_guide.md) for detailed documentation.

## Loading Configuration

### From YAML File

```python
from pygsquig.io import load_config

# Load configuration
config = load_config("simulation.yml")

# Use with run.py script
# pygsquig-run simulation.yml --device=gpu
```

### From Python Dictionary

```python
from pygsquig.io.config import RunConfig

config_dict = {
    'grid': {'N': 512, 'L': 2 * np.pi},
    'solver': {
        'alpha': 1.0,
        'dissipation': {'type': 'hyperviscosity', 'nu_p': 1e-8, 'p': 8}
    },
    'forcing': {
        'type': 'ring',
        'kf': 40.0,
        'epsilon': 0.1
    },
    'simulation': {
        't_end': 100.0,
        'output_interval': 1.0
    }
}

config = RunConfig.from_dict(config_dict)
```

### Programmatic Configuration

```python
from pygsquig.io.simple_config import SimpleConfig

# Create config programmatically
config = SimpleConfig()
config.set_grid(N=512, L=2*np.pi)
config.set_physics(alpha=1.0, nu_p=1e-8, p=8)
config.set_forcing('ring', kf=40.0, epsilon=0.1)
config.set_time_integration(dt=0.001, t_final=100.0)
config.set_output('simulation_output.h5', save_interval=1.0)

# Run
solver = config.create_solver()
results = config.run_simulation(solver)
```

## Configuration Templates

### Decaying Turbulence

```yaml
# decaying_turbulence.yml
grid:
  N: 1024
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-10
    p: 8
  time_integration:
    method: RK4
    adaptive_cfl: true
    cfl_safety: 0.8

initial_condition:
  type: random
  seed: 42
  amplitude: 1.0

simulation:
  t_end: 50.0
  output_interval: 1.0
  checkpoint_interval: 10.0

output:
  fields: [theta, velocity]
  diagnostics: [energy_spectrum, enstrophy]
```

### Forced-Dissipative Turbulence

```yaml
# forced_turbulence.yml
grid:
  N: 512
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-8
    p: 8
  damping:
    type: linear_drag
    mu: 0.1
    k_cutoff_factor: 0.5

forcing:
  type: ring
  kf: 40.0
  dk: 2.0
  epsilon: 0.1
  tau_f: 0.0

simulation:
  t_end: 200.0
  output_interval: 10.0
  log_interval: 1.0

output:
  fields: [theta]
  diagnostics: [energy_spectrum, scalar_flux, enstrophy]
```

### Simulation with Passive Scalars

```yaml
# scalar_mixing.yml
grid:
  N: 256
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-8
    p: 8

forcing:
  type: ring
  kf: 30.0
  epsilon: 0.1

scalars:
  enabled: true
  species:
    - name: "tracer"
      kappa: 0.001
      initial_condition: "gaussian"
      initial_params:
        center: [3.14159, 3.14159]
        width: 0.5
    
    - name: "temperature"
      kappa: 0.01
      source:
        type: "localized"
        parameters:
          x0: 1.57
          y0: 3.14159
          sigma: 0.3
          amplitude: 1.0
      initial_condition: "uniform"
      initial_params:
        value: 20.0

simulation:
  t_end: 100.0
  output_interval: 1.0

output:
  fields: [theta, scalars]
  diagnostics: [energy_spectrum, scalar_flux]
```

### High-Resolution Production Run

```yaml
# production_run.yml
grid:
  N: 2048
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-12
    p: 16  # Higher order for scale separation
  time_integration:
    method: RK4
    adaptive_cfl: true
    cfl_safety: 0.7
    dt_max: 0.001

forcing:
  type: ring
  kf: 100.0
  epsilon: 0.1

simulation:
  t_end: 1000.0
  output_interval: 10.0
  checkpoint_interval: 100.0
  wall_time_limit: 86400  # 24 hours

output:
  fields: [theta]
  diagnostics: [energy_spectrum, enstrophy]
  compress: true
```

## Best Practices

1. **Start Simple**: Begin with low resolution and short times
2. **Check Stability**: Monitor CFL and energy conservation
3. **Scale Gradually**: Increase resolution in steps
4. **Save Smartly**: Balance output frequency with storage
5. **Use Checkpoints**: For long production runs
6. **Profile First**: Test performance before production

## Validation

Configuration validation checks:
- Grid resolution is even
- CFL safety < 1
- Wavenumbers within resolved range
- Compatible parameter combinations
- Output directory exists

## Environment Variables

Override config with environment variables:

```bash
# Override device
export PYGSQUIG_DEVICE=gpu:1

# Set precision
export JAX_ENABLE_X64=1

# Logging level
export PYGSQUIG_LOG_LEVEL=DEBUG
```
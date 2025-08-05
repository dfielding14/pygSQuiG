# Passive Scalar Module Documentation

## Overview

The passive scalar module (`pygsquig.scalars`) provides functionality for evolving passive scalars advected by gSQG velocity fields. This enables studying mixing, transport, and reactive processes in turbulent flows.

## Mathematical Formulation

The module solves the advection-diffusion equation with sources:

```
∂θ_p/∂t + u·∇θ_p = κ∇²θ_p + S(θ_p, x, t)
```

where:
- `θ_p` is the passive scalar field
- `u` is the velocity field from the gSQG dynamics
- `κ` is the molecular diffusivity
- `S` is an optional source term

## Quick Start

### Basic Usage

```python
from pygsquig.core.grid import make_grid
from pygsquig.core.solver_with_scalars import gSQGSolverWithScalars
from pygsquig.scalars.source_terms import LocalizedSource

# Create grid
grid = make_grid(256, 2*np.pi)

# Define passive scalars
passive_scalars = {
    'dye': {'kappa': 0.01},  # Simple dye with diffusion
    'temperature': {
        'kappa': 0.02,
        'source': LocalizedSource(
            amplitude=1.0,
            x0=np.pi, y0=np.pi,
            sigma=0.5
        )
    }
}

# Create solver with scalars
solver = gSQGSolverWithScalars(
    grid, alpha=1.0, nu_p=1e-6, p=8,
    passive_scalars=passive_scalars
)

# Initialize
state = solver.initialize(
    seed=42,  # Random active scalar
    scalar_init={
        'dye': np.ones((256, 256)),
        'temperature': np.zeros((256, 256))
    }
)

# Step forward
state = solver.step(state, dt=0.01)
```

### Using the PassiveScalarEvolver Directly

```python
from pygsquig.scalars import PassiveScalarEvolver, ExponentialGrowth

# Create evolver for a single scalar
evolver = PassiveScalarEvolver(
    grid=grid,
    kappa=0.01,
    source_fn=ExponentialGrowth(rate=-0.1),  # Decay
    name="tracer"
)

# Initialize
scalar_state = evolver.initialize(scalar0=initial_field)

# Step with given velocity
scalar_state = evolver.step(scalar_state, dt=0.01, u=u_field, v=v_field)
```

## Source Terms

### Built-in Source Terms

1. **ExponentialGrowth**: `S = λ * θ_p`
   ```python
   source = ExponentialGrowth(rate=0.1)  # Growth
   source = ExponentialGrowth(rate=-0.1)  # Decay
   ```

2. **LocalizedSource**: `S = A * exp(-r²/σ²)`
   ```python
   source = LocalizedSource(
       amplitude=1.0,
       x0=np.pi, y0=np.pi,  # Center location
       sigma=0.5            # Width
   )
   ```

3. **ChemicalReaction**: `S = -k * θ_p²`
   ```python
   source = ChemicalReaction(
       rate=1.0,
       threshold=0.1  # Optional: only react above threshold
   )
   ```

4. **TimePeriodicSource**: `S = A * sin(ωt + φ) * f(x,y)`
   ```python
   source = TimePeriodicSource(
       amplitude=1.0,
       frequency=2*np.pi,  # ω
       phase=0.0          # φ
   )
   ```

### Custom Source Terms

Create custom sources by inheriting from `SourceTerm`:

```python
from pygsquig.scalars.source_terms import SourceTerm

class CustomSource(SourceTerm):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, scalar, grid, t):
        # Return source in physical space
        return self.param1 * scalar + self.param2 * np.sin(grid.x)
```

## Diagnostics

### Available Diagnostics

```python
from pygsquig.scalars.diagnostics import *

# Variance spectrum
k_bins, spectrum = compute_scalar_variance_spectrum(scalar_hat, grid)

# Turbulent flux <u'θ'>
flux_x, flux_y = compute_scalar_flux(scalar_hat, u, v, grid)

# Dissipation rate χ = κ<|∇θ|²>
chi = compute_scalar_dissipation(scalar_hat, grid, kappa)

# Mixing efficiency
efficiency = compute_mixing_efficiency(
    scalar_initial, scalar_final, grid, time_elapsed
)

# PDF moments (mean, variance, skewness, kurtosis)
moments = compute_scalar_pdf_moments(scalar_hat, max_moment=4)
```

### Extracting Diagnostics from Solver

```python
# Get all diagnostics including scalars
diagnostics = solver.get_diagnostics(state)

# Scalar-specific diagnostics have prefixed names
print(f"Dye mean: {diagnostics['dye_mean']}")
print(f"Temperature variance: {diagnostics['temperature_variance']}")
```

## Multi-Species Evolution

For coupled systems with multiple scalars:

```python
from pygsquig.scalars import MultiSpeciesEvolver

# Define species
species = {
    'nutrient': {'kappa': 0.01, 'source': decay_source},
    'phytoplankton': {'kappa': 0.001, 'source': growth_source},
    'zooplankton': {'kappa': 0.001}
}

# Create multi-species evolver
evolver = MultiSpeciesEvolver(grid, species)

# Add coupled source (e.g., predator-prey)
def coupled_source(scalars_dict, grid, t):
    N = scalars_dict['nutrient']
    P = scalars_dict['phytoplankton']
    Z = scalars_dict['zooplankton']
    
    return {
        'nutrient': -0.1 * N * P,           # Nutrient uptake
        'phytoplankton': 0.1 * N * P - 0.05 * P * Z,  # Growth - grazing
        'zooplankton': 0.03 * P * Z - 0.01 * Z        # Grazing - mortality
    }

evolver.coupled_sources.append(coupled_source)
```

## Performance Considerations

1. **JAX JIT Compilation**: All core operations are JIT-compiled for performance
2. **Batch Processing**: Multiple scalars are evolved together efficiently
3. **Memory Usage**: Each scalar requires O(N²) memory
4. **GPU Acceleration**: Automatically uses GPU if available

## Example Applications

### 1. Mixing Study
```python
# Initialize with step function
dye = np.where(grid.x < np.pi, 1.0, 0.0)

# Measure mixing rate
initial_variance = np.var(dye)
# ... evolve ...
mixing_rate = (initial_variance - final_variance) / time_elapsed
```

### 2. Reaction-Diffusion
```python
# Chemical reaction A + B -> C
def reaction_source(scalars, grid, t):
    A = scalars['A']
    B = scalars['B']
    rate = 0.1
    reaction = -rate * A * B
    
    return {
        'A': reaction,
        'B': reaction,
        'C': -2 * reaction  # Production
    }
```

### 3. Biological Dynamics
```python
# Phytoplankton with light-dependent growth
class PhytoplanktonGrowth(SourceTerm):
    def __init__(self, growth_rate, light_profile):
        self.growth_rate = growth_rate
        self.light = light_profile
        
    def __call__(self, P, grid, t):
        # Growth depends on light and nutrients
        return self.growth_rate * P * self.light(grid.y) * (1 - P)
```

## Integration with Existing Code

To add passive scalars to existing simulations:

1. Replace `gSQGSolver` with `gSQGSolverWithScalars`
2. Define passive scalars in constructor
3. Initialize with `scalar_init` parameter
4. Everything else works the same!

The extended solver maintains full backward compatibility - it can be used as a drop-in replacement even without scalars.

## Validation

The module includes comprehensive tests in `tests/test_passive_scalar.py`:
- Conservation properties
- Diffusion accuracy
- Source term verification
- Numerical stability

Run tests with:
```bash
pytest tests/test_passive_scalar.py
```
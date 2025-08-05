# Passive Scalars Guide

This guide covers the passive scalar functionality in pygSQuiG, which allows you to simulate the advection-diffusion of passive tracers alongside the main gSQG dynamics.

## Overview

Passive scalars are quantities that are transported by the flow but do not affect the flow dynamics. Common examples include:
- Temperature in incompressible flows
- Chemical concentrations
- Dyes and tracers
- Pollutants
- Biological quantities

The passive scalar equation solved by pygSQuiG is:

```
∂θ/∂t + u·∇θ = κ∇²θ + S(θ, x, y, t)
```

where:
- `θ` is the passive scalar field
- `u` is the velocity field from the gSQG dynamics
- `κ` is the scalar diffusivity
- `S` is an optional source term

## Configuration

### Basic Setup

To enable passive scalars in your simulation, add a `scalars` section to your configuration:

```yaml
scalars:
  enabled: true
  species:
    - name: "tracer"
      kappa: 0.001
      initial_condition: "random"
      initial_params:
        seed: 42
        amplitude: 0.1
```

### Multiple Species

You can simulate multiple passive scalar species simultaneously:

```yaml
scalars:
  enabled: true
  species:
    - name: "temperature"
      kappa: 0.01
      initial_condition: "gaussian"
      initial_params:
        center: [3.14159, 3.14159]
        width: 1.0
    
    - name: "salinity"
      kappa: 0.005
      initial_condition: "uniform"
      initial_params:
        value: 35.0
    
    - name: "dye"
      kappa: 0.001
      initial_condition: "zero"
```

### Initial Conditions

Available initial condition types:

1. **Zero**: `initial_condition: "zero"`
   - Scalar field initialized to zero everywhere

2. **Uniform**: `initial_condition: "uniform"`
   ```yaml
   initial_params:
     value: 1.0  # Constant value
   ```

3. **Random**: `initial_condition: "random"`
   ```yaml
   initial_params:
     seed: 42        # Random seed
     amplitude: 0.1  # RMS amplitude
   ```

4. **Gaussian**: `initial_condition: "gaussian"`
   ```yaml
   initial_params:
     center: [3.14159, 3.14159]  # Center position
     width: 1.0                  # Gaussian width
   ```

## Source Terms

Passive scalars can include source terms that add, remove, or transform the scalar quantity.

### Exponential Growth/Decay

Models exponential growth (positive rate) or decay (negative rate):

```yaml
source:
  type: "exponential"
  parameters:
    rate: -0.1  # Decay rate (1/time)
```

The source term is: `S = rate × θ`

### Localized Source

Adds scalar at a specific location with a Gaussian profile:

```yaml
source:
  type: "localized"
  parameters:
    x0: 3.14159      # x-coordinate of source center
    y0: 3.14159      # y-coordinate of source center
    sigma: 0.5       # Gaussian width
    amplitude: 1.0   # Source strength
```

### Chemical Reaction

Models second-order chemical decay:

```yaml
source:
  type: "chemical"
  parameters:
    rate: 0.1         # Reaction rate
    threshold: 0.01   # Optional: minimum concentration for reaction
```

The source term is: `S = -rate × θ²`

### Time-Periodic Source

Adds scalar with time-periodic modulation:

```yaml
source:
  type: "periodic"
  parameters:
    amplitude: 1.0     # Source amplitude
    frequency: 6.28    # Angular frequency (rad/time)
    phase: 0.0        # Phase shift
```

## Output and Diagnostics

### Saving Scalar Fields

To save scalar fields in the output, include "scalars" in the output fields:

```yaml
output:
  fields: [theta, velocity, scalars]
  diagnostics: [energy_spectrum, scalar_flux]
```

Scalar fields will be saved with names like `scalar_temperature`, `scalar_dye`, etc.

### Scalar Diagnostics

When scalars are enabled, additional diagnostics are computed:

- `{name}_mean`: Mean value of the scalar
- `{name}_variance`: Variance of the scalar  
- `{name}_max`: Maximum value
- `{name}_min`: Minimum value
- `{name}_total`: Integrated total amount
- `{name}_source_integrated`: Cumulative source contribution
- `{name}_dissipation_integrated`: Cumulative dissipation

These are automatically saved to the diagnostics file.

## Examples

### Example 1: Passive Tracer in Turbulence

Study mixing of a passive tracer in forced turbulence:

```yaml
# Grid and solver configuration
grid:
  N: 256
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-8
    p: 8

# Forcing
forcing:
  type: ring
  kf: 30.0
  dk: 2.0
  epsilon: 0.1

# Passive scalars
scalars:
  enabled: true
  species:
    - name: "tracer"
      kappa: 0.001
      initial_condition: "gaussian"
      initial_params:
        center: [3.14159, 3.14159]
        width: 0.5

# Output
output:
  fields: [theta, scalars]
  diagnostics: [energy_spectrum, scalar_flux]
```

### Example 2: Reacting Scalars

Simulate chemical species with reactions:

```yaml
scalars:
  enabled: true
  species:
    - name: "reactant_A"
      kappa: 0.01
      source:
        type: "chemical"
        parameters:
          rate: 0.1
      initial_condition: "uniform"
      initial_params:
        value: 1.0
    
    - name: "reactant_B"
      kappa: 0.01
      source:
        type: "localized"
        parameters:
          x0: 1.57
          y0: 3.14
          sigma: 0.3
          amplitude: 0.5
      initial_condition: "zero"
```

### Example 3: Temperature with Heating and Cooling

Model temperature evolution with localized heating and uniform cooling:

```yaml
scalars:
  enabled: true
  species:
    - name: "temperature"
      kappa: 0.02
      source:
        type: "localized"
        parameters:
          x0: 3.14159
          y0: 3.14159
          sigma: 0.5
          amplitude: 10.0  # Heating
      initial_condition: "uniform"
      initial_params:
        value: 20.0  # Initial temperature
```

## Using Scalars in Python

### Basic Usage

```python
from pygsquig.core.solver_with_scalars import gSQGSolverWithScalars
from pygsquig.scalars.source_terms import ExponentialGrowth

# Define passive scalars
passive_scalars = {
    'dye': {
        'kappa': 0.001,
        'source': ExponentialGrowth(rate=-0.1)
    }
}

# Create solver with scalars
solver = gSQGSolverWithScalars(
    grid=grid,
    alpha=1.0,
    nu_p=1e-8,
    p=8,
    passive_scalars=passive_scalars
)

# Initialize with scalar fields
scalar_init = {'dye': np.ones((N, N))}
state = solver.initialize(seed=42, scalar_init=scalar_init)

# Run simulation
for step in range(n_steps):
    state = solver.step(state, dt)
```

### Custom Source Terms

Create custom source terms by subclassing `SourceTerm`:

```python
from pygsquig.scalars.source_terms import SourceTerm
import jax.numpy as jnp

class CustomSource(SourceTerm):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, scalar, grid, t):
        # Return source in physical space
        source = self.param1 * jnp.sin(self.param2 * grid.x)
        return source
```

### Accessing Scalar Data

```python
# During simulation
if hasattr(state, 'scalar_state'):
    for name, scalar_hat in state.scalar_state.scalars.items():
        scalar_phys = ifft2(scalar_hat).real
        print(f"{name}: mean={scalar_phys.mean()}")

# From saved output
import xarray as xr
ds = xr.open_dataset("output/fields_00001000.nc")
temperature = ds["scalar_temperature"].values
```

## Advanced Features

### Adaptive Scalar Diffusivity

For Schmidt number studies, scale diffusivity with viscosity:

```python
# Schmidt number Sc = ν/κ
Sc = 1.0  # Prandtl number for temperature
kappa = solver.nu_p / Sc
```

### Coupled Source Terms

For scalars that interact (not yet fully implemented):

```python
def coupled_reaction(scalars_dict, grid, t):
    A = scalars_dict['A']
    B = scalars_dict['B']
    # A + B -> C
    rate = 0.1
    reaction = -rate * A * B
    return {'A': reaction, 'B': reaction, 'C': -reaction}
```

### Scalar Variance Spectrum

Analyze scalar mixing at different scales:

```python
from pygsquig.scalars.diagnostics import compute_scalar_variance_spectrum

k_bins, spectrum = compute_scalar_variance_spectrum(
    scalar_hat, grid
)
plt.loglog(k_bins, spectrum)
plt.xlabel('k')
plt.ylabel('Scalar variance spectrum')
```

## Performance Considerations

1. **Memory**: Each scalar field requires `N²` complex numbers in spectral space
2. **Computation**: Each scalar adds one RHS evaluation per timestep
3. **I/O**: Scalar fields increase output file sizes proportionally

### Optimization Tips

- Use fewer scalars when possible
- Consider coarser resolution for scalars than velocity
- Save scalar fields less frequently than velocity
- Use single precision for large-scale studies

## Common Issues

### Numerical Instability

If scalars become NaN or overflow:
- Reduce timestep
- Increase scalar diffusivity
- Check source term parameters
- Ensure initial conditions are reasonable

### Mass Conservation

For conservative scalars (no sources):
- Total scalar should be conserved
- Check using diagnostics: `{name}_total`
- Small drift indicates timestep issues

### Boundary Conditions

pygSQuiG uses periodic boundary conditions. For localized sources near boundaries:
- Sources wrap around due to periodicity
- Place sources away from boundaries if unwanted

## References

- Lapeyre (2017): "Surface quasi-geostrophy"
- Smith & Ferrari (2009): "The production and dissipation of compensated thermohaline variance"
- Sukhatme & Pierrehumbert (2002): "Decay of passive scalars under the action of single scale smooth velocity fields"
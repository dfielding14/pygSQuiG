# Forcing Patterns Guide

This guide covers all available forcing patterns in pygSQuiG for driving turbulence and maintaining statistically stationary states.

## Overview

Forcing is essential for:
- Maintaining turbulence against dissipation
- Studying energy cascades
- Driving specific flow patterns
- Investigating response to perturbations

## Forcing Categories

### 1. Stochastic Forcing

Random forcing patterns that maintain statistical properties.

#### Ring Forcing (Most Common)
```python
from pygsquig.forcing.ring_forcing import RingForcing

forcing = RingForcing(
    kf=30.0,        # Central wavenumber
    dk=2.0,         # Bandwidth (modes in [kf-dk, kf+dk])
    epsilon=0.1,    # Energy injection rate
    tau_f=0.0       # Correlation time (0 = white noise)
)
```

**Use cases**:
- Homogeneous isotropic turbulence
- Controlled energy injection scale
- Well-defined cascade ranges

#### White Noise Forcing
```python
from pygsquig.forcing.stochastic_forcing import WhiteNoiseForcing

forcing = WhiteNoiseForcing(
    amplitude=0.5,
    k_min=20.0,     # Minimum wavenumber
    k_max=40.0,     # Maximum wavenumber
    isotropy=True   # Maintain isotropy
)
```

**Properties**:
- Uncorrelated in time
- Uniform in spectral band
- No preferential direction

#### Colored Noise Forcing
```python
from pygsquig.forcing.stochastic_forcing import ColoredNoiseForcing

forcing = ColoredNoiseForcing(
    amplitude=0.3,
    spectral_slope=-2.0,  # Red noise
    k_peak=5.0,          # Peak wavenumber
    k_width=3.0          # Spectral width
)
```

**Applications**:
- Large-scale atmospheric forcing
- Ocean mesoscale dynamics
- Non-white spectral injection

#### Ornstein-Uhlenbeck Forcing
```python
from pygsquig.forcing.stochastic_forcing import OrnsteinUhlenbeckForcing

forcing = OrnsteinUhlenbeckForcing(
    amplitude=0.2,
    correlation_time=1.0,  # Temporal correlation
    k_min=10.0,
    k_max=30.0
)
```

**Features**:
- Temporal correlations
- Smooth forcing evolution
- Realistic for slow forcing

### 2. Deterministic Forcing

Coherent patterns for studying specific phenomena.

#### Taylor-Green Forcing
```python
from pygsquig.forcing.deterministic_forcing import TaylorGreenForcing

forcing = TaylorGreenForcing(
    amplitude=1.0,
    k=4,            # Wavenumber (integer)
    time_dependence='steady'  # or 'oscillatory', 'growing'
)
```

**Physics**:
- Classic vortex pattern
- Symmetric flow structure
- Tests numerical accuracy

#### Kolmogorov Forcing
```python
from pygsquig.forcing.deterministic_forcing import KolmogorovForcing

forcing = KolmogorovForcing(
    amplitude=1.0,
    k_y=4,          # Forcing wavenumber in y
    time_dependence='steady'
)
```

**Applications**:
- Unidirectional forcing
- Studies of anisotropy
- Zonal flow generation

#### Checkerboard Forcing
```python
from pygsquig.forcing.deterministic_forcing import CheckerboardForcing

forcing = CheckerboardForcing(
    amplitude=1.0,
    k=8,            # Checkerboard wavenumber
    time_dependence='steady'
)
```

**Properties**:
- Alternating sign pattern
- No net momentum input
- Mixing studies

#### ABC (Arnold-Beltrami-Childress) Forcing
```python
from pygsquig.forcing.deterministic_forcing import ABCForcing

forcing = ABCForcing(
    A=1.0, B=0.7, C=0.5,  # ABC coefficients
    k=4,                   # Base wavenumber
    time_dependence='steady'
)
```

**Features**:
- 3D-like complexity in 2D
- Chaotic trajectories
- Helicity analog

### 3. Physical Forcing Patterns

Geophysically relevant forcing.

#### Shear Layer Forcing
```python
from pygsquig.forcing.physical_forcing import ShearLayerForcing

forcing = ShearLayerForcing(
    amplitude=1.0,
    shear_width=0.1,      # Width of shear layer (fraction of L)
    n_layers=2,           # Number of shear layers
    orientation='horizontal',  # or 'vertical'
    time_dependence='steady'
)
```

**Applications**:
- Kelvin-Helmholtz instability
- Mixing layer dynamics
- Atmospheric jets

#### Jet Forcing
```python
from pygsquig.forcing.physical_forcing import JetForcing

forcing = JetForcing(
    amplitude=2.0,
    jet_width=0.15,       # Jet width (fraction of L)
    n_jets=3,             # Number of jets
    meander_amplitude=0.1, # Jet meandering
    meander_k=2,          # Meandering wavenumber
    orientation='zonal'    # or 'meridional'
)
```

**Physics**:
- Geostrophic jets
- Baroclinic dynamics
- Storm track studies

#### Vortex Injection
```python
from pygsquig.forcing.stochastic_forcing import StochasticVortexForcing

forcing = StochasticVortexForcing(
    amplitude=2.0,
    vortex_size=0.1,      # Size relative to domain
    injection_rate=1.0,   # Vortices per unit time
    vortex_strength_std=0.3
)
```

**Use cases**:
- Convective plumes
- Coherent structure injection
- Vortex interactions

### 4. Combined Forcing

Mix multiple forcing types:

```python
from pygsquig.forcing.stochastic_forcing import create_combined_stochastic_forcing
from pygsquig.forcing.combined_forcing import CombinedForcing

# Stochastic combination
stochastic_combined = create_combined_stochastic_forcing(
    [white_noise, colored_noise, vortex_forcing],
    weights=[1.0, 2.0, 0.5]
)

# General combination (deterministic + stochastic)
combined = CombinedForcing(
    forcings=[ring_forcing, jet_forcing, shear_forcing],
    weights=[1.0, 0.5, 0.3],
    combination_type='additive'  # or 'multiplicative'
)
```

## Implementation Details

### Forcing Interface

All forcing classes implement:
```python
def __call__(self, theta_hat, key, dt, grid):
    """
    Args:
        theta_hat: Current field in spectral space
        key: JAX random key (for stochastic)
        dt: Time step
        grid: Grid object
    
    Returns:
        forcing_hat: Forcing in spectral space
    """
```

### Time Modulation

Many forcings support time dependence:
```python
forcing = TaylorGreenForcing(
    amplitude=1.0,
    k=4,
    time_dependence='oscillatory',
    omega=2.0,  # Oscillation frequency
    phase=0.0   # Initial phase
)
```

Options:
- `'steady'`: Constant amplitude
- `'oscillatory'`: `A(t) = A₀ cos(ωt + φ)`
- `'growing'`: `A(t) = A₀ exp(γt)`
- `'decaying'`: `A(t) = A₀ exp(-γt)`
- `'pulsed'`: Periodic pulses

### Energy Control

Set forcing to achieve target energy flux:

```python
# For ring forcing
epsilon_target = 0.1  # Desired energy injection
kf = 30.0            # Forcing scale
n_modes = 2 * np.pi * kf * dk  # Approximate forced modes

# Amplitude for target injection
amplitude = np.sqrt(2 * epsilon_target / n_modes)

forcing = RingForcing(kf=kf, dk=dk, epsilon=epsilon_target)
```

## Usage Examples

### Example 1: Inverse Cascade Study
```python
# Large-scale forcing for inverse cascade
forcing = RingForcing(
    kf=50.0,        # Small-scale forcing
    dk=2.0,
    epsilon=0.1
)

# Run simulation
solver = gSQGSolver(grid, alpha=0.0)  # 2D Euler
state = solver.initialize(seed=42)

for step in range(10000):
    state = solver.step(state, dt, forcing=forcing, key=key)
```

### Example 2: Jet-Turbulence Interaction
```python
# Combine jets with turbulent forcing
jet_forcing = JetForcing(
    amplitude=2.0,
    jet_width=0.1,
    n_jets=2,
    orientation='zonal'
)

turb_forcing = RingForcing(
    kf=40.0,
    epsilon=0.05
)

combined = CombinedForcing(
    forcings=[jet_forcing, turb_forcing],
    weights=[1.0, 0.5]
)
```

### Example 3: Shear-Driven Turbulence
```python
# Shear layer with perturbations
shear = ShearLayerForcing(
    amplitude=1.0,
    shear_width=0.05,
    n_layers=1
)

noise = WhiteNoiseForcing(
    amplitude=0.01,  # Small perturbations
    k_min=20,
    k_max=60
)

combined = CombinedForcing([shear, noise], [1.0, 0.1])
```

## Choosing Forcing Parameters

### For Cascade Studies
- **Inverse cascade**: Force at small scales (high k)
- **Forward cascade**: Force at large scales (low k)
- **Dual cascade**: Force at intermediate scales

### For Stationary Turbulence
Balance energy injection with dissipation:
```python
# Estimate dissipation scale
k_d = (epsilon / nu_p^3)^(1/(3p-2))

# Force well below dissipation scale
kf = k_d / 4
```

### For Instability Studies
- Start with steady deterministic forcing
- Add small noise perturbations
- Observe growth rates and patterns

## Advanced Features

### Custom Forcing

Create your own forcing:

```python
from pygsquig.forcing.base import BaseForcing

class MyCustomForcing(BaseForcing):
    def __init__(self, params):
        self.params = params
    
    def __call__(self, theta_hat, key, dt, grid):
        # Implement forcing logic
        forcing_hat = ...
        return forcing_hat
```

### Adaptive Forcing

Adjust forcing based on flow state:

```python
class AdaptiveForcing(BaseForcing):
    def __call__(self, theta_hat, key, dt, grid):
        # Measure current energy
        energy = np.sum(np.abs(theta_hat)**2) / grid.N**4
        
        # Adjust amplitude
        target_energy = 1.0
        amplitude = self.base_amplitude * (target_energy / energy)
        
        # Apply forcing
        return amplitude * self.pattern
```

## Visualization

Visualize forcing patterns:

```python
import matplotlib.pyplot as plt

# Get forcing in physical space
forcing_hat = forcing(theta_hat, key, dt, grid)
forcing_phys = grid.ifft2(forcing_hat).real

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(forcing_phys, cmap='RdBu_r', extent=[0, L, 0, L])
plt.colorbar(label='Forcing')
plt.title(f'{forcing.__class__.__name__}')
plt.show()
```

## Performance Tips

1. **Pre-compute patterns**: For deterministic forcing
2. **Use JIT**: Wrap forcing calls with `jax.jit`
3. **Vectorize**: Process multiple forcings together
4. **Memory**: Reuse arrays when possible

## Troubleshooting

### Energy Not Stationary
- Check forcing amplitude vs dissipation
- Verify wavenumber selection
- Monitor energy flux

### Numerical Instability  
- Forcing too strong for timestep
- Check CFL with forcing velocity
- Use adaptive timestepping

### Anisotropic Results
- Verify forcing isotropy
- Check grid resolution
- Use appropriate averaging
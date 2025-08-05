# Core Concepts

Understanding the physics and numerical methods behind pygSQuiG will help you design better simulations and interpret results correctly.

## The Generalized SQG Equations

### Mathematical Formulation

The generalized Surface Quasi-Geostrophic (gSQG) equations describe the evolution of a scalar field θ (potential temperature or buoyancy):

```
∂θ/∂t + u·∇θ = F - D
```

where the velocity field is related to θ through:

```
u = ∇⊥ψ,    ψ = -(-Δ)^(-α/2) θ
```

Here:
- `θ` is the active scalar (temperature/buoyancy)
- `u = (u, v)` is the velocity field
- `ψ` is the stream function
- `α` is the fractional exponent (0 ≤ α ≤ 2)
- `F` is forcing
- `D` is dissipation
- `∇⊥ = (-∂/∂y, ∂/∂x)` is the perpendicular gradient

### Special Cases

The parameter α determines the physics:

| α | System | Relationship | Physics |
|---|--------|--------------|---------|
| 0 | 2D Euler | ψ = -Δ^(-1) θ | Vorticity dynamics |
| 1 | SQG | ψ = -(-Δ)^(-1/2) θ | Surface buoyancy |
| 2 | Linear | ψ = -θ | Passive scalar |

### Energy and Enstrophy

The system conserves (in the inviscid, unforced limit):
- **Energy**: `E = ½⟨θ²⟩`
- **Generalized enstrophy**: `Z = ½⟨|(-Δ)^(α/4) θ|²⟩`

## Numerical Methods

### Pseudo-Spectral Method

pygSQuiG uses Fourier spectral methods for spatial discretization:

1. **Physical to spectral**: `θ̂ = FFT(θ)`
2. **Derivatives in spectral space**: `∂θ̂/∂x = ik_x θ̂`
3. **Nonlinear terms**: Computed in physical space
4. **Spectral to physical**: `θ = IFFT(θ̂)`

### Dealiasing

To prevent aliasing errors from nonlinear terms:
- **2/3 rule**: Zero out modes with `|k| > (2/3)k_max`
- **Padding**: Alternative method (not used by default)

### Time Integration

#### RK4 (Default)
Fourth-order Runge-Kutta:
```python
k1 = f(y_n, t_n)
k2 = f(y_n + dt/2 * k1, t_n + dt/2)
k3 = f(y_n + dt/2 * k2, t_n + dt/2)
k4 = f(y_n + dt * k3, t_n + dt)
y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

#### Adaptive Timestepping
CFL-based automatic timestep adjustment:
```python
dt = CFL_safety * min(dx/max|u|, dx^(2p)/nu_p)
```

### Hyperviscosity

High-order dissipation to remove energy at small scales:
```
D = ν_p (-Δ)^p θ
```

Typical values:
- `p = 4`: Fourth-order (∇⁸)
- `p = 8`: Eighth-order (∇¹⁶)
- `ν_p`: Adjusted for resolution

## Grid and Spectral Space

### Physical Grid
- Domain: `[0, L] × [0, L]` (doubly periodic)
- Grid points: `N × N` (must be even)
- Grid spacing: `dx = L/N`

### Wavenumber Space
- Wavenumbers: `k = 2π n/L`, `n = -N/2, ..., N/2-1`
- Maximum resolved: `k_max = π N/L`
- Dealiased maximum: `k_dealias = (2/3) k_max`

### Fourier Conventions

pygSQuiG uses NumPy/JAX FFT conventions:
```python
# Forward transform
F[k] = Σ f[x] exp(-2πi k·x/L)

# Inverse transform  
f[x] = (1/N²) Σ F[k] exp(2πi k·x/L)
```

Energy conservation: `⟨|f|²⟩ = (1/N²) Σ|F[k]|²`

## Physical Regimes

### Inverse Cascade (2D Turbulence, α < 2/3)

Energy flows to large scales:
- Energy spectrum: `E(k) ~ k^(-5/3)`
- Enstrophy cascade to small scales
- Formation of large coherent vortices

### Forward Cascade (SQG, α = 1)

Energy cascades to small scales:
- Energy spectrum: `E(k) ~ k^(-5/3)`
- Surface dynamics
- Filament formation and frontogenesis

### Transitional Regimes (2/3 < α < 1)

Mixed behavior with both cascades possible.

## Forcing and Dissipation

### Energy Balance

In statistical equilibrium:
```
dE/dt = ε - ε_d = 0
```
where:
- `ε`: Energy injection rate (forcing)
- `ε_d`: Energy dissipation rate

### Forcing Types

1. **Ring forcing**: Narrow band in k-space
2. **Deterministic**: Coherent patterns (Taylor-Green, etc.)
3. **Stochastic**: Random (white/colored noise)
4. **Physical**: Vortex/jet injection

### Dissipation Mechanisms

1. **Hyperviscosity**: `-ν_p (∇²)^p θ`
2. **Linear drag**: `-r θ` (Rayleigh friction)
3. **Selective damping**: Large-scale removal
4. **Sponge layers**: Near boundaries

## Passive Scalars

Additional fields advected by the flow:
```
∂θ_p/∂t + u·∇θ_p = κ∇²θ_p + S
```

Properties:
- No feedback on velocity
- Same advection as active scalar
- Independent diffusion and sources
- Useful for studying mixing

## Dimensionless Parameters

Key non-dimensional numbers:

### Reynolds Number
```
Re = U L / ν
```
- `U`: Characteristic velocity
- `L`: Domain size
- `ν`: Effective viscosity

### Froude Number (for SQG)
```
Fr = U / (N H)
```
- `N`: Buoyancy frequency
- `H`: Layer depth

### Péclet Number (for scalars)
```
Pe = U L / κ
```
- `κ`: Scalar diffusivity

## Best Practices

### Resolution Requirements

Ensure adequate resolution:
```python
# Dissipation scale
k_d = (ε / ν_p^3)^(1/(3p-2))

# Need k_max > 2 * k_d for accuracy
N_min = 2 * k_d * L / π
```

### Timestep Selection

For stability:
```python
# CFL condition
dt_cfl = 0.5 * dx / max|u|

# Viscous condition
dt_visc = 0.25 * dx^(2p) / ν_p

dt = min(dt_cfl, dt_visc)
```

### Energy Injection

Match forcing to desired regime:
```python
# Estimate velocity from energy injection
U ~ (ε L)^(1/3)

# Set forcing amplitude
forcing_amplitude = ε / (n_forced_modes * τ_corr)
```

## Diagnostics and Analysis

### Essential Diagnostics

1. **Energy**: `E = ½⟨θ²⟩`
2. **Enstrophy**: `Z = ½⟨|∇θ|²⟩`
3. **Dissipation**: `ε_d = ν_p⟨|∇^p θ|²⟩`
4. **Energy flux**: `Π(k) = -⟨θ̂*(k) N̂L(k)⟩`

### Spectral Analysis

Energy spectrum computation:
```python
E(k) = ½ Σ_{|k'|∈[k-δk/2,k+δk/2]} |θ̂(k')|²
```

### Structure Functions

For intermittency analysis:
```python
S_p(r) = ⟨|θ(x+r) - θ(x)|^p⟩
```

## Common Pitfalls

1. **Under-resolution**: Insufficient grid points for cascade
2. **Wrong timestep**: CFL violations or inefficiency
3. **Forcing imbalance**: Too strong/weak injection
4. **Aliasing**: Not using dealiasing
5. **Boundary effects**: In non-periodic domains

## Further Reading

- Held et al. (1995): Original SQG paper
- Constantin et al. (1994): Mathematical properties
- Pierrehumbert et al. (2000): Spectra and cascades
- Sukhatme & Pierrehumbert (2002): Decay dynamics
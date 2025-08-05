# pygSQuiG Next Steps - Development Plan

## Current Status (2025-08-05)
- ✅ Core numerics fully implemented and tested (138 tests passing)
- ✅ JAX pytree integration and JIT optimization complete
- ✅ Performance profiling done - identified bottlenecks and created optimized solver
- ✅ Comprehensive documentation, tutorials, and examples in place
- ✅ Validation suite confirms correct physics

## Implementation Order (Start Here Tomorrow)

1. **Day 1-2**: Start with passive scalar evolution (Phase 1)
   - Implement core `PassiveScalarEvolver` class
   - Add basic source terms (exponential, localized)
   - Integrate with main solver
   - Write tests

2. **Day 3**: Complete passive scalars and start deterministic forcing
   - Finish passive scalar diagnostics
   - Implement Kolmogorov and Taylor-Green forcing
   - Test pattern formation

3. **Day 4-5**: GPU optimization
   - Profile current GPU performance
   - Implement single-GPU optimizations
   - Start multi-GPU support

4. **Day 6**: Adaptive timestepping
   - Implement CFL-based dt control
   - Integrate with existing solver
   - Test stability and efficiency

## High Priority Tasks

### 1. Passive Scalar Evolution with Source Terms
**Goal**: Add capability to evolve passive scalars advected by the gSQG velocity field with user-defined source terms

**Scientific Motivation**:
- Study mixing and transport in turbulent flows
- Model reactive tracers and chemical species
- Investigate scalar variance cascade
- Enable comparison with experimental dye/temperature measurements
- Foundation for studying:
  - Phytoplankton dynamics in ocean turbulence
  - Atmospheric pollutant dispersion
  - Combustion and reaction-diffusion systems
  - Magnetic field evolution (passive vector extension)
  - Temperature/salinity in ocean dynamics

**Key Scientific Applications**:
1. **Mixing efficiency**: Quantify how turbulence enhances mixing
2. **Reaction rates**: Study how turbulence affects chemical reactions
3. **Biological models**: Nutrient uptake, plankton patchiness
4. **Climate models**: Tracer transport in atmosphere/ocean
5. **Industrial applications**: Combustion, chemical reactors

**Implementation Plan**:

1. **Core Passive Scalar Module** (`pygsquig/core/passive_scalar.py`):
   ```python
   class PassiveScalarEvolver:
       """Evolves passive scalars: ∂_t θ_p + u·∇θ_p = κ∇²θ_p + S"""
       
       def __init__(self, 
                    grid: Grid,
                    kappa: float = 0.0,  # Diffusivity
                    source_fn: Optional[Callable] = None):
           self.grid = grid
           self.kappa = kappa
           self.source_fn = source_fn
           
       def compute_rhs(self, 
                      theta_p_hat: jax.Array,
                      u: jax.Array, 
                      v: jax.Array,
                      t: float = 0.0) -> jax.Array:
           """Compute RHS for passive scalar evolution"""
           # Physical space operations
           theta_p = ifft2(theta_p_hat)
           
           # Advection: -u·∇θ_p
           advection = -jacobian(theta_p, u, v, self.grid)
           advection_hat = fft2(advection)
           
           # Diffusion: κ∇²θ_p
           diffusion_hat = -self.kappa * self.grid.k2 * theta_p_hat
           
           # Source term (in physical or spectral space)
           source_hat = jnp.zeros_like(theta_p_hat)
           if self.source_fn is not None:
               source = self.source_fn(theta_p, self.grid, t)
               source_hat = fft2(source) if source.shape == theta_p.shape else source
               
           return advection_hat + diffusion_hat + source_hat
   ```

2. **Source Term Interface**:
   ```python
   # Base class for source terms
   class SourceTerm:
       """Abstract base for passive scalar source terms"""
       def __call__(self, theta_p: jax.Array, grid: Grid, t: float) -> jax.Array:
           raise NotImplementedError
   
   # Example implementations
   class ExponentialGrowth(SourceTerm):
       """S = λ * θ_p (exponential growth/decay)"""
       def __init__(self, rate: float):
           self.rate = rate
           
       def __call__(self, theta_p: jax.Array, grid: Grid, t: float) -> jax.Array:
           return self.rate * theta_p
   
   class LocalizedSource(SourceTerm):
       """S = A * exp(-r²/σ²) (Gaussian source)"""
       def __init__(self, amplitude: float, x0: float, y0: float, sigma: float):
           self.A = amplitude
           self.x0, self.y0 = x0, y0
           self.sigma2 = sigma**2
           
       def __call__(self, theta_p: jax.Array, grid: Grid, t: float) -> jax.Array:
           r2 = (grid.x - self.x0)**2 + (grid.y - self.y0)**2
           return self.A * jnp.exp(-r2 / self.sigma2)
   
   class ChemicalReaction(SourceTerm):
       """S = -k * θ_p² (quadratic decay)"""
       def __init__(self, rate: float):
           self.k = rate
           
       def __call__(self, theta_p: jax.Array, grid: Grid, t: float) -> jax.Array:
           return -self.k * theta_p**2
   ```

3. **Integration with Main Solver**:
   - Extend state dictionary to include passive scalars
   - Modify solver to optionally evolve passive scalars
   ```python
   # In gSQGSolver
   def __init__(self, ..., passive_scalars: Optional[Dict[str, PassiveScalarEvolver]] = None):
       self.passive_scalars = passive_scalars or {}
   
   def initialize_with_scalars(self, theta0=None, scalar_init: Dict[str, jax.Array] = None):
       state = self.initialize(theta0)
       if scalar_init:
           state['scalars'] = {name: fft2(field) for name, field in scalar_init.items()}
       return state
   
   def step(self, state, dt, ...):
       # Step active scalar
       theta_hat_new = ... # existing code
       
       # Compute velocity for passive scalar advection
       u, v = self.compute_velocity(theta_hat_new)
       
       # Step each passive scalar
       if 'scalars' in state and self.passive_scalars:
           scalars_new = {}
           for name, scalar_evolver in self.passive_scalars.items():
               scalar_hat = state['scalars'][name]
               rhs_fn = lambda s: scalar_evolver.compute_rhs(s, u, v, state['time'])
               scalars_new[name] = rk4_step(scalar_hat, rhs_fn, dt)
           state['scalars'] = scalars_new
       
       return state
   ```

4. **Multi-Species Infrastructure** (Phase 2):
   ```python
   class MultiSpeciesEvolver:
       """Evolve multiple coupled passive scalars"""
       
       def __init__(self, 
                    grid: Grid,
                    species: Dict[str, Dict[str, Any]]):
           """
           species = {
               'temperature': {'kappa': 0.01, 'source': HeatSource()},
               'salinity': {'kappa': 0.001, 'source': None},
               'dye': {'kappa': 0.01, 'source': DyeRelease()}
           }
           """
           self.evolvers = {}
           for name, config in species.items():
               self.evolvers[name] = PassiveScalarEvolver(
                   grid, 
                   kappa=config['kappa'],
                   source_fn=config.get('source')
               )
       
       def add_coupled_source(self, source_fn: Callable):
           """Add source that depends on multiple species"""
           self.coupled_sources.append(source_fn)
   ```

5. **Diagnostics and Analysis**:
   - Extend diagnostics to include scalar statistics
   - Add scalar variance spectrum computation
   - Implement scalar-velocity correlation diagnostics
   ```python
   def compute_scalar_variance_spectrum(scalar_hat, grid):
       """Compute spectrum of scalar variance"""
       return compute_energy_spectrum(scalar_hat, grid, alpha=0)
   
   def compute_scalar_flux(scalar_hat, u, v, grid):
       """Compute turbulent scalar flux <u'θ'>"""
       scalar = ifft2(scalar_hat)
       flux_x = jnp.mean(u * scalar)
       flux_y = jnp.mean(v * scalar)
       return flux_x, flux_y
   ```

6. **Example Use Cases**:
   ```python
   # Example 1: Passive dye with exponential decay
   dye_evolver = PassiveScalarEvolver(
       grid, 
       kappa=0.01,
       source_fn=ExponentialGrowth(rate=-0.1)  # Decay
   )
   
   # Example 2: Temperature with localized heating
   temp_evolver = PassiveScalarEvolver(
       grid,
       kappa=0.02,
       source_fn=LocalizedSource(amplitude=1.0, x0=np.pi, y0=np.pi, sigma=0.5)
   )
   
   # Create solver with passive scalars
   solver = gSQGSolver(
       grid, alpha=1.0, nu_p=1e-4,
       passive_scalars={
           'dye': dye_evolver,
           'temperature': temp_evolver
       }
   )
   
   # Initialize with scalar fields
   state = solver.initialize_with_scalars(
       theta0=initial_vorticity,
       scalar_init={
           'dye': np.ones((N, N)),  # Uniform initial dye
           'temperature': gaussian_blob  # Localized temperature
       }
   )
   ```

**Testing Requirements**:
- Verify advection-diffusion equation solved correctly
- Test various source terms (growth, decay, localized)
- Check conservation properties (with no source)
- Validate against analytical solutions where available
- Test multi-species coupling

**Performance Considerations**:
- Batch operations for multiple scalars
- Reuse velocity field computation
- Option for lower resolution scalars
- GPU optimization for scalar operations

### 2. Deterministic Forcing Implementation
**Goal**: Add deterministic forcing patterns for studying specific flow configurations

**Implementation Plan**:
1. Create `pygsquig/forcing/deterministic_forcing.py`
2. Implement key forcing patterns:
   ```python
   class KolmogorovForcing:
       """F = A * sin(ky * y) - classic pattern for turbulence studies"""
       def __init__(self, amplitude: float, wavenumber: int)
       
   class TaylorGreenForcing:
       """F = A * sin(kx*x) * sin(ky*y) - vortex array forcing"""
       def __init__(self, amplitude: float, kx: int, ky: int)
       
   class ShearForcing:
       """F = A * sin(ky * y) * δ(kx) - creates mean shear flow"""
       def __init__(self, amplitude: float, ky: int)
       
   class CustomForcing:
       """User-defined forcing from a function"""
       def __init__(self, forcing_fn: Callable)
   ```
3. Ensure all forcing classes:
   - Follow same interface as RingForcing
   - Work in spectral space
   - Are JIT-compatible
   - Include get_diagnostics() method
4. Add tests in `tests/test_deterministic_forcing.py`
5. Create example configs:
   - `configs/kolmogorov_flow.yml`
   - `configs/taylor_green.yml`
6. Add notebook demonstrating pattern formation

### 2. GPU Optimization and Multi-GPU Support
**Goal**: Maximize performance on GPU hardware, enable large-scale simulations

**Implementation Plan**:
1. **GPU Profiling**:
   - Create `pygsquig/scripts/profile_gpu.py`
   - Use JAX profiler to identify GPU bottlenecks
   - Test memory transfer patterns
   - Benchmark against CPU at various resolutions

2. **Single GPU Optimizations**:
   - Optimize memory layout for coalesced access
   - Investigate XLA compilation flags
   - Test mixed precision (float32 vs float64)
   - Optimize FFT plans for GPU

3. **Multi-GPU Implementation**:
   - Create `pygsquig/core/solver_parallel.py`
   - Use `jax.pmap` for data parallelism
   - Implement domain decomposition:
     ```python
     @partial(jax.pmap, axis_name='device')
     def parallel_step(state, dt, device_id):
         # Each device handles part of the domain
     ```
   - Handle halo exchanges for spectral operations
   - Test weak and strong scaling

4. **Integration**:
   - Add `n_devices` parameter to solver
   - Auto-detect available GPUs
   - Fallback to single GPU/CPU gracefully
   - Update run.py to support multi-GPU

5. **Testing**:
   - Verify results match single GPU
   - Benchmark scaling efficiency
   - Test on different GPU architectures

### 3. Adaptive Timestepping with CFL Control
**Goal**: Optimize time integration for stability and efficiency

**Implementation Plan**:
1. **CFL Monitoring**:
   - Enhance existing `compute_cfl()` in time_integrator.py
   - Add spectral CFL for hyperviscosity
   - Monitor both advective and diffusive CFL

2. **Adaptive Algorithm**:
   ```python
   class AdaptiveTimestepper:
       def __init__(self, cfl_target=0.8, dt_min=1e-6, dt_max=0.1):
           # Safety factors and limits
           
       def compute_dt(self, state, solver):
           # Compute various CFL numbers
           cfl_adv = compute_advective_cfl(u, v, dx, dy)
           cfl_diff = compute_diffusive_cfl(nu_p, p, k_max)
           
           # Adjust dt to meet target CFL
           dt_new = dt * (cfl_target / max(cfl_adv, cfl_diff))
           return np.clip(dt_new, dt_min, dt_max)
   ```

3. **Integration with Solver**:
   - Add `adaptive_dt` option to solver
   - Implement timestep history tracking
   - Add rejection/retry logic for stability
   - Ensure compatibility with checkpointing

4. **Error-based Control** (optional):
   - Implement embedded RK schemes (e.g., RK45)
   - Add local error estimation
   - Combined CFL/error control

5. **Testing**:
   - Verify conservation properties maintained
   - Test on stiff problems (high hyperviscosity)
   - Benchmark efficiency vs fixed dt
   - Test with forcing/damping

## Medium Priority Tasks

### 4. White Noise Forcing
- Implement δ-correlated in time stochastic forcing
- Add to RingForcing or create WhiteNoiseForcing class
- Include spatial correlation options
- Test statistical properties

### 5. Docker Container
- Create GPU-enabled Docker image
- Include Jupyter environment
- Add example data and configs
- Document deployment process

## Low Priority Tasks

### 6. IMEX Time Integration
- For very stiff hyperviscosity
- Implement Crank-Nicolson for diffusion
- Explicit treatment of advection

### 7. Enhanced Documentation
- Theory guide with derivations
- More advanced tutorials
- Performance tuning guide
- Contributing guidelines

## Technical Notes

### JAX Best Practices to Follow:
- Use `jax.lax` operations where possible
- Avoid Python loops in hot paths
- Careful with random number handling in pmap
- Profile before optimizing

### Testing Strategy:
- Each new feature needs comprehensive tests
- Maintain >90% test coverage
- Add integration tests for new components
- Benchmark performance improvements

### Backwards Compatibility:
- New features should not break existing API
- Optional parameters for new functionality
- Clear deprecation path if needed

## Timeline Estimate
- Passive scalar evolution: 3-4 days (including multi-species infrastructure)
- Deterministic forcing: 2-3 days
- GPU optimization: 3-4 days  
- Adaptive timestepping: 2-3 days
- White noise & Docker: 2 days
- Documentation & testing: Ongoing

## Priority Rationale

1. **Passive Scalar Evolution** - Placed first because:
   - Significantly extends scientific capabilities
   - Enables new classes of problems (mixing, reactions, transport)
   - Foundation for multi-physics simulations
   - High demand in turbulence research community

2. **Deterministic Forcing** - Essential for:
   - Studying specific flow patterns
   - Comparing with classical results
   - Pattern formation studies
   - Turbulence transition research

3. **GPU Optimization** - Critical for:
   - Large-scale simulations
   - Parameter studies
   - Real research applications
   - Computational efficiency

4. **Adaptive Timestepping** - Important for:
   - Stability with minimal computational cost
   - Handling stiff problems
   - Production runs
   - User convenience

## File Structure for New Components
```
pygsquig/
├── forcing/
│   ├── deterministic_forcing.py  # NEW
│   └── white_noise.py           # NEW
├── core/
│   ├── passive_scalar.py        # NEW - PassiveScalarEvolver, SourceTerm classes
│   ├── solver_parallel.py       # NEW - Multi-GPU support
│   └── adaptive_timestepper.py  # NEW - CFL-based adaptive dt
├── scalars/                      # NEW directory
│   ├── __init__.py
│   ├── source_terms.py          # NEW - Library of source terms
│   └── diagnostics.py           # NEW - Scalar-specific diagnostics
└── scripts/
    ├── profile_gpu.py            # NEW
    └── test_passive_scalar.py    # NEW - Validation scripts
```

## Key Success Metrics
1. **Passive Scalar Evolution**:
   - Correctly solves advection-diffusion equation (validated against analytical solutions)
   - Source terms produce expected behavior (exponential growth, localized heating, reactions)
   - Multi-species evolution maintains conservation properties
   - Scalar variance cascade follows theoretical predictions
   - Performance overhead <20% for single scalar, <10% per additional scalar

2. **Deterministic Forcing**:
   - Kolmogorov flow produces correct steady state
   - Taylor-Green vortices evolve as expected
   - Pattern formation matches published results
   
3. **GPU Optimization**:
   - Single GPU: >5x speedup for N≥512
   - Multi-GPU: >80% scaling efficiency on 4 GPUs
   - Memory usage optimized for large grids
   
4. **Adaptive Timestepping**:
   - Reduces computation time by >30% for typical runs
   - Maintains stability for stiff problems
   - Preserves conservation properties
   
5. **Overall**:
   - All features maintain numerical accuracy
   - Comprehensive test coverage (>90%)
   - Clear documentation and examples
   - Backwards compatibility maintained
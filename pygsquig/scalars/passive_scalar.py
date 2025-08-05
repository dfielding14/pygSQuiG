"""
Core passive scalar evolution module.

This module provides the PassiveScalarEvolver class for evolving
passive scalars advected by velocity fields with diffusion and sources.
"""

from typing import Optional, Callable, Tuple, Dict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.core.operators import jacobian
from pygsquig.core.time_integrator import rk4_step
from pygsquig.scalars.state import PassiveScalarState, MultiScalarState
from pygsquig.scalars.source_terms import SourceTerm
from pygsquig.validation import validate_diffusivity
from pygsquig.exceptions import PassiveScalarError


@jax.jit
def compute_scalar_advection(
    scalar: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, grid: Grid
) -> jnp.ndarray:
    """Compute advection term for passive scalar (JIT-compiled).

    Computes -u·∇θ using the Jacobian function.

    Args:
        scalar: Scalar field in physical space
        u: x-velocity component in physical space
        v: y-velocity component in physical space
        grid: Grid object

    Returns:
        Advection term in spectral space
    """
    # Compute -u·∇θ (negative because jacobian computes J(θ,ψ) = ∂θ/∂x ∂ψ/∂y - ∂θ/∂y ∂ψ/∂x)
    advection = -jacobian(scalar, u, v, grid)
    return fft2(advection)


@jax.jit
def compute_scalar_diffusion(scalar_hat: jnp.ndarray, grid: Grid, kappa: float) -> jnp.ndarray:
    """Compute diffusion term for passive scalar (JIT-compiled).

    Computes κ∇²θ in spectral space.

    Args:
        scalar_hat: Scalar field in spectral space
        grid: Grid object
        kappa: Diffusivity coefficient

    Returns:
        Diffusion term in spectral space
    """
    # Diffusion: κ∇²θ = -κk²θ̂
    return -kappa * grid.k2 * scalar_hat


@jax.jit
def compute_passive_scalar_rhs(
    scalar_hat: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: Grid,
    kappa: float,
    source_hat: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute full RHS for passive scalar evolution (JIT-compiled).

    Computes ∂θ/∂t = -u·∇θ + κ∇²θ + S

    Args:
        scalar_hat: Scalar field in spectral space
        u: x-velocity in physical space
        v: y-velocity in physical space
        grid: Grid object
        kappa: Diffusivity coefficient
        source_hat: Source term in spectral space (optional)

    Returns:
        RHS in spectral space
    """
    # Convert to physical space for advection
    scalar = ifft2(scalar_hat)

    # Advection term
    advection_hat = compute_scalar_advection(scalar, u, v, grid)

    # Diffusion term
    diffusion_hat = compute_scalar_diffusion(scalar_hat, grid, kappa)

    # Combine terms
    rhs = advection_hat + diffusion_hat

    # Add source if provided
    if source_hat is not None:
        rhs = rhs + source_hat

    return rhs


@dataclass
class PassiveScalarEvolver:
    """Evolves a passive scalar field advected by a velocity field.

    Solves: ∂θ/∂t + u·∇θ = κ∇²θ + S

    Where:
        - θ is the passive scalar
        - u is the advecting velocity field
        - κ is the diffusivity
        - S is an optional source term

    Attributes:
        grid: Grid object
        kappa: Diffusivity coefficient
        source_fn: Optional source term function
        name: Name identifier for the scalar
    """

    grid: Grid
    kappa: float = 0.0
    source_fn: Optional[SourceTerm] = None
    name: str = "scalar"

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.kappa = validate_diffusivity(self.kappa, "kappa")

        if self.source_fn is not None and not isinstance(self.source_fn, SourceTerm):
            raise PassiveScalarError(
                f"source_fn must be a SourceTerm instance, " f"got {type(self.source_fn).__name__}"
            )

    def initialize(
        self, scalar0: Optional[jnp.ndarray] = None, seed: Optional[int] = None
    ) -> PassiveScalarState:
        """Initialize passive scalar state.

        Args:
            scalar0: Initial scalar field in physical space
            seed: Random seed for generating initial condition

        Returns:
            Initial PassiveScalarState

        Raises:
            PassiveScalarError: If neither scalar0 nor seed provided
        """
        if scalar0 is not None:
            scalar_hat = fft2(scalar0)
        elif seed is not None:
            # Random initial condition
            key = jax.random.PRNGKey(seed)
            scalar = jax.random.normal(key, shape=(self.grid.N, self.grid.N))
            scalar_hat = fft2(scalar)
            # Smooth by killing high wavenumbers
            k_cutoff = self.grid.N // 4
            mask = jnp.sqrt(self.grid.k2) < k_cutoff * 2 * jnp.pi / self.grid.L
            scalar_hat = scalar_hat * mask
        else:
            raise PassiveScalarError("Must provide either scalar0 or seed for initialization")

        return PassiveScalarState(
            scalar_hat=scalar_hat, time=0.0, total_source=0.0, total_dissipation=0.0
        )

    def compute_source(self, scalar_hat: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute source term in spectral space.

        Args:
            scalar_hat: Current scalar field in spectral space
            t: Current time

        Returns:
            Source term in spectral space
        """
        if self.source_fn is None:
            return jnp.zeros_like(scalar_hat)

        # Compute source in physical space
        scalar = ifft2(scalar_hat)
        source = self.source_fn(scalar, self.grid, t)

        # Transform to spectral space
        return fft2(source)

    def compute_rhs(
        self, scalar_hat: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, t: float = 0.0
    ) -> jnp.ndarray:
        """Compute RHS for passive scalar evolution.

        Args:
            scalar_hat: Scalar field in spectral space
            u: x-velocity in physical space
            v: y-velocity in physical space
            t: Current time

        Returns:
            RHS in spectral space
        """
        # Compute source term if present
        if self.source_fn is not None:
            source_hat = self.compute_source(scalar_hat, t)
        else:
            source_hat = None

        # Use JIT-compiled RHS function
        return compute_passive_scalar_rhs(scalar_hat, u, v, self.grid, self.kappa, source_hat)

    def step(
        self, state: PassiveScalarState, dt: float, u: jnp.ndarray, v: jnp.ndarray
    ) -> PassiveScalarState:
        """Advance passive scalar by one timestep.

        Args:
            state: Current passive scalar state
            dt: Timestep
            u: x-velocity in physical space
            v: y-velocity in physical space

        Returns:
            Updated PassiveScalarState
        """

        # Define RHS function for time integrator
        def rhs_fn(scalar_hat):
            return self.compute_rhs(scalar_hat, u, v, state.time)

        # Time step using RK4
        scalar_hat_new = rk4_step(state.scalar_hat, rhs_fn, dt)

        # Update diagnostics
        if self.source_fn is not None:
            # Compute integrated source
            source = self.compute_source(state.scalar_hat, state.time)
            scalar = ifft2(state.scalar_hat)
            source_phys = ifft2(source)
            total_source = state.total_source + dt * jnp.mean(scalar * source_phys)
        else:
            total_source = state.total_source

        # Compute integrated dissipation
        if self.kappa > 0:
            dissipation = compute_scalar_diffusion(state.scalar_hat, self.grid, self.kappa)
            scalar = ifft2(state.scalar_hat)
            diss_phys = ifft2(dissipation)
            total_dissipation = state.total_dissipation + dt * jnp.mean(scalar * diss_phys)
        else:
            total_dissipation = state.total_dissipation

        return PassiveScalarState(
            scalar_hat=scalar_hat_new,
            time=state.time + dt,
            total_source=float(total_source),
            total_dissipation=float(total_dissipation),
        )

    def get_diagnostics(self, state: PassiveScalarState) -> Dict[str, float]:
        """Compute diagnostic quantities for the scalar field.

        Args:
            state: Current scalar state

        Returns:
            Dictionary of diagnostics
        """
        scalar = ifft2(state.scalar_hat)

        # Basic statistics
        mean_value = float(jnp.mean(scalar))
        variance = float(jnp.var(scalar))
        max_value = float(jnp.max(scalar))
        min_value = float(jnp.min(scalar))

        # Total scalar (should be conserved without sources)
        total_scalar = float(jnp.sum(scalar) * (self.grid.L / self.grid.N) ** 2)

        return {
            f"{self.name}_mean": mean_value,
            f"{self.name}_variance": variance,
            f"{self.name}_max": max_value,
            f"{self.name}_min": min_value,
            f"{self.name}_total": total_scalar,
            f"{self.name}_source_integrated": state.total_source,
            f"{self.name}_dissipation_integrated": state.total_dissipation,
        }


@dataclass
class MultiSpeciesEvolver:
    """Evolves multiple passive scalars simultaneously.

    This class manages multiple PassiveScalarEvolver instances
    and can handle coupled source terms.

    Attributes:
        grid: Grid object
        species: Dictionary of species configurations
        coupled_sources: List of coupled source functions
    """

    grid: Grid
    species: Dict[str, Dict] = field(default_factory=dict)
    coupled_sources: list = field(default_factory=list)

    def __post_init__(self):
        """Create evolvers for each species."""
        self.evolvers = {}

        for name, config in self.species.items():
            self.evolvers[name] = PassiveScalarEvolver(
                grid=self.grid,
                kappa=config.get("kappa", 0.0),
                source_fn=config.get("source", None),
                name=name,
            )

    def initialize(self, initial_fields: Dict[str, jnp.ndarray]) -> MultiScalarState:
        """Initialize all scalar fields.

        Args:
            initial_fields: Dictionary mapping species names to initial fields

        Returns:
            MultiScalarState with all species initialized
        """
        scalar_states = {}

        for name, evolver in self.evolvers.items():
            if name in initial_fields:
                state = evolver.initialize(scalar0=initial_fields[name])
            else:
                # Default to zero initial condition
                state = evolver.initialize(scalar0=jnp.zeros((self.grid.N, self.grid.N)))
            scalar_states[name] = state.scalar_hat

        return MultiScalarState(scalars=scalar_states, time=0.0, diagnostics={})

    def step(
        self, state: MultiScalarState, dt: float, u: jnp.ndarray, v: jnp.ndarray
    ) -> MultiScalarState:
        """Advance all scalars by one timestep.

        Args:
            state: Current multi-scalar state
            dt: Timestep
            u: x-velocity in physical space
            v: y-velocity in physical space

        Returns:
            Updated MultiScalarState
        """
        new_scalars = {}
        new_diagnostics = {}

        # Step each scalar
        for name, evolver in self.evolvers.items():
            # Create temporary state for this scalar
            scalar_state = PassiveScalarState(scalar_hat=state.scalars[name], time=state.time)

            # Step forward
            new_scalar_state = evolver.step(scalar_state, dt, u, v)

            # Store results
            new_scalars[name] = new_scalar_state.scalar_hat
            new_diagnostics[name] = evolver.get_diagnostics(new_scalar_state)

        # Apply coupled sources if any
        if self.coupled_sources:
            new_scalars = self._apply_coupled_sources(new_scalars, state.time + dt, dt)

        return MultiScalarState(
            scalars=new_scalars, time=state.time + dt, diagnostics=new_diagnostics
        )

    def _apply_coupled_sources(
        self, scalars: Dict[str, jnp.ndarray], t: float, dt: float
    ) -> Dict[str, jnp.ndarray]:
        """Apply coupled source terms.

        Args:
            scalars: Current scalar fields in spectral space
            t: Current time
            dt: Timestep

        Returns:
            Updated scalar fields
        """
        # Convert to physical space
        scalars_phys = {name: ifft2(scalar_hat) for name, scalar_hat in scalars.items()}

        # Apply each coupled source
        for source_fn in self.coupled_sources:
            sources = source_fn(scalars_phys, self.grid, t)

            # Add source contributions
            for name, source in sources.items():
                if name in scalars:
                    source_hat = fft2(source)
                    scalars[name] = scalars[name] + dt * source_hat

        return scalars

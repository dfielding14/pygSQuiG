"""
Main solver for the generalized Surface Quasi-Geostrophic equations.

This module provides the core solver class that orchestrates the simulation
of gSQG turbulence using spectral methods.
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.core.operators import (
    compute_velocity_from_theta,
    hyperviscosity,
    jacobian,
    laplacian,
)
from pygsquig.core.time_integrator import rk4_step

# Type alias for state
State = Dict[str, jax.Array]


# Note: hyperviscosity function is now imported from operators module


@jax.jit
def _compute_core_rhs(
    theta_hat: jax.Array, grid: Grid, alpha: float, nu_p: float, p: int
) -> jax.Array:
    """
    Compute core RHS without forcing/damping (JIT-compiled).

    Parameters:
        theta_hat: Current state in spectral space
        grid: Grid object
        alpha: Fractional power
        nu_p: Hyperviscosity coefficient
        p: Hyperviscosity order

    Returns:
        Core RHS (advection + dissipation)
    """
    # Get theta in physical space
    theta = ifft2(theta_hat)

    # Compute velocity
    u, v = compute_velocity_from_theta(theta_hat, grid, alpha)

    # Compute advection term: -u·∇θ
    advection = -jacobian(theta, u, v, grid)
    advection_hat = fft2(advection)

    # Hyperviscous dissipation
    dissipation_hat = hyperviscosity(theta_hat, grid, nu_p, p)

    return advection_hat + dissipation_hat


class gSQGSolver:
    """
    Solver for generalized Surface Quasi-Geostrophic equations.

    Solves: ∂_t θ + u·∇θ = F - D
    where: u = ∇^⊥(-Δ)^(-α/2)θ

    Attributes:
        grid: Grid object containing spatial discretization
        alpha: Fractional power in velocity relation (α ∈ [-2, 2])
        nu_p: Hyperviscosity coefficient
        p: Hyperviscosity order (p=2,4,8)
    """

    def __init__(self, grid: Grid, alpha: float, nu_p: float = 0.0, p: int = 8):
        """
        Initialize gSQG solver.

        Parameters:
            grid: Grid object
            alpha: Fractional power for gSQG dynamics
            nu_p: Hyperviscosity coefficient (default 0)
            p: Hyperviscosity order (default 8)
        """
        self.grid = grid
        self.alpha = alpha
        self.nu_p = nu_p
        self.p = p

        # Validate parameters
        if not -2 <= alpha <= 2:
            raise ValueError(f"alpha must be in [-2, 2], got {alpha}")
        if p not in [2, 4, 8]:
            raise ValueError(f"p must be 2, 4, or 8, got {p}")

    def initialize(self, theta0: Optional[jax.Array] = None, seed: Optional[int] = None) -> State:
        """
        Initialize simulation state.

        Parameters:
            theta0: Initial condition for θ in physical space (optional)
            seed: Random seed for random initial condition (optional)

        Returns:
            Initial state dictionary
        """
        if theta0 is not None:
            # Use provided initial condition
            theta_hat = fft2(theta0)
        elif seed is not None:
            # Generate random initial condition
            key = jax.random.PRNGKey(seed)
            theta = jax.random.normal(key, shape=(self.grid.N, self.grid.N))
            # Apply some smoothing
            theta_hat = fft2(theta)
            # Kill high wavenumbers for smooth start
            k_cutoff = self.grid.N // 8
            mask = jnp.sqrt(self.grid.k2) < k_cutoff * 2 * jnp.pi / self.grid.L
            theta_hat = theta_hat * mask
        else:
            # Default: zero initial condition
            theta_hat = jnp.zeros((self.grid.N, self.grid.N), dtype=jnp.complex128)

        return {"theta_hat": theta_hat, "time": jnp.array(0.0), "step": jnp.array(0)}

    def compute_velocity(self, theta_hat: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Compute velocity field from θ.

        Parameters:
            theta_hat: Fourier coefficients of θ

        Returns:
            (u, v): Velocity components in physical space
        """
        return compute_velocity_from_theta(theta_hat, self.grid, self.alpha)

    def compute_hyperviscosity(self, theta_hat: jax.Array) -> jax.Array:
        """
        Compute hyperviscous dissipation term.

        Parameters:
            theta_hat: Fourier coefficients of θ

        Returns:
            Hyperviscous term -νp(-Δ)^(p/2)θ in spectral space
        """
        return hyperviscosity(theta_hat, self.grid, self.nu_p, self.p)

    def compute_rhs(
        self,
        theta_hat: jax.Array,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        **kwargs,
    ) -> jax.Array:
        """
        Compute right-hand side of gSQG equation.

        Parameters:
            theta_hat: Current state in spectral space
            forcing: Optional forcing function (not used in basic solver)
            damping: Optional damping function (not used in basic solver)
            **kwargs: Additional arguments for forcing/damping

        Returns:
            Time derivative of θ in spectral space
        """
        # Compute core RHS (JIT-compiled)
        rhs = _compute_core_rhs(theta_hat, self.grid, self.alpha, self.nu_p, self.p)

        # Add forcing if provided
        if forcing is not None:
            rhs = rhs + forcing(theta_hat, **kwargs)

        # Add damping if provided
        if damping is not None:
            rhs = rhs + damping(theta_hat, **kwargs)

        return rhs

    def step(
        self,
        state: State,
        dt: float,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        **kwargs,
    ) -> State:
        """
        Advance solution by one time step.

        Parameters:
            state: Current state dictionary
            dt: Time step size
            forcing: Optional forcing function
            damping: Optional damping function
            **kwargs: Additional arguments (e.g., PRNGKey for forcing)

        Returns:
            Updated state dictionary
        """
        # Extract current theta
        theta_hat = state["theta_hat"]

        # Define RHS function for time integrator
        def rhs_fn(theta_hat_):
            return self.compute_rhs(theta_hat_, forcing, damping, **kwargs)

        # Time step using RK4
        theta_hat_new = rk4_step(theta_hat, rhs_fn, dt)

        # Update state
        return {"theta_hat": theta_hat_new, "time": state["time"] + dt, "step": state["step"] + 1}

    def get_diagnostics(self, state: State) -> Dict[str, float]:
        """
        Compute diagnostic quantities.

        Parameters:
            state: Current state

        Returns:
            Dictionary of diagnostic values
        """
        theta_hat = state["theta_hat"]
        theta = ifft2(theta_hat)

        # Kinetic energy
        u, v = self.compute_velocity(theta_hat)
        ke = 0.5 * jnp.mean(u**2 + v**2)

        # Enstrophy (squared θ)
        enstrophy = 0.5 * jnp.mean(theta**2)

        # Maximum vorticity
        max_theta = jnp.max(jnp.abs(theta))

        return {
            "kinetic_energy": float(ke),
            "enstrophy": float(enstrophy),
            "max_theta": float(max_theta),
            "time": float(state["time"]),
            "step": int(state["step"]),
        }

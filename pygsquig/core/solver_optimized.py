"""
Optimized solver for the generalized Surface Quasi-Geostrophic equations.

This module provides an optimized version of the gSQG solver that uses
JAX-specific optimizations for better performance.
"""

from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.core.operators import (
    compute_velocity_from_theta,
    jacobian,
)
from pygsquig.core.time_integrator import rk4_step

# Type alias for state
State = dict[str, jax.Array]


@jax.jit
def _compute_core_rhs_optimized(
    theta_hat: jax.Array, grid: Grid, alpha: float, nu_p: float, p: int
) -> jax.Array:
    """
    Optimized core RHS computation with fused operations.
    """
    # Compute velocity and advection in one go
    theta = ifft2(theta_hat)
    u, v = compute_velocity_from_theta(theta_hat, grid, alpha)

    # Fused advection and FFT
    advection = -jacobian(theta, u, v, grid)
    advection_hat = fft2(advection)

    # Fused dissipation
    k2_pow = grid.k2 ** (p / 2)
    dissipation_hat = -nu_p * k2_pow * theta_hat

    return advection_hat + dissipation_hat


@jax.jit
def _rk4_step_optimized(
    theta_hat: jax.Array, grid: Grid, alpha: float, nu_p: float, p: float, dt: float
) -> jax.Array:
    """
    Optimized RK4 step with parameters.
    """

    # RK4 stages
    k1 = _compute_core_rhs_optimized(theta_hat, grid, alpha, nu_p, p)
    k2 = _compute_core_rhs_optimized(theta_hat + 0.5 * dt * k1, grid, alpha, nu_p, p)
    k3 = _compute_core_rhs_optimized(theta_hat + 0.5 * dt * k2, grid, alpha, nu_p, p)
    k4 = _compute_core_rhs_optimized(theta_hat + dt * k3, grid, alpha, nu_p, p)

    return theta_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _multistep_scan(
    carry: tuple[jax.Array, float],
    _: None,
    grid: Grid,
    alpha: float,
    nu_p: float,
    p: float,
    dt: float,
) -> tuple[tuple[jax.Array, float], jax.Array]:
    """
    Single step function for lax.scan multistep integration.
    """
    theta_hat, time = carry
    theta_hat_new = _rk4_step_optimized(theta_hat, grid, alpha, nu_p, p, dt)
    time_new = time + dt
    return (theta_hat_new, time_new), theta_hat_new


class gSQGSolverOptimized:
    """
    Optimized solver for generalized Surface Quasi-Geostrophic equations.

    Key optimizations:
    - Fused operations to reduce memory traffic
    - lax.scan for multi-step integration
    - Bundled parameters for better JIT compilation
    - Pre-computed arrays for dissipation
    """

    def __init__(self, grid: Grid, alpha: float, nu_p: float = 0.0, p: int = 8):
        """Initialize optimized gSQG solver."""
        self.grid = grid
        self.alpha = alpha
        self.nu_p = nu_p
        self.p = p

        # Validate parameters
        if not -2 <= alpha <= 2:
            raise ValueError(f"alpha must be in [-2, 2], got {alpha}")
        if p not in [2, 4, 8]:
            raise ValueError(f"p must be 2, 4, or 8, got {p}")

        # Bundle parameters for JIT
        self.params = {"alpha": alpha, "nu_p": nu_p, "p": float(p)}

        # Pre-compute dissipation array for better performance
        self.dissipation_factor = nu_p * (grid.k2 ** (p / 2))

    def initialize(self, theta0: Optional[jax.Array] = None, seed: Optional[int] = None) -> State:
        """Initialize simulation state (same as base solver)."""
        if theta0 is not None:
            theta_hat = fft2(theta0)
        elif seed is not None:
            key = jax.random.PRNGKey(seed)
            theta = jax.random.normal(key, shape=(self.grid.N, self.grid.N))
            theta_hat = fft2(theta)
            # Kill high wavenumbers for smooth start
            k_cutoff = self.grid.N // 8
            mask = jnp.sqrt(self.grid.k2) < k_cutoff * 2 * jnp.pi / self.grid.L
            theta_hat = theta_hat * mask
        else:
            theta_hat = jnp.zeros((self.grid.N, self.grid.N), dtype=jnp.complex128)

        return {"theta_hat": theta_hat, "time": jnp.array(0.0), "step": jnp.array(0)}

    @partial(jax.jit, static_argnums=(0, 2))
    def step(
        self,
        state: State,
        dt: float,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        **kwargs,
    ) -> State:
        """
        Optimized single time step.

        Note: For best performance with forcing/damping, consider using
        multistep integration instead.
        """
        theta_hat = state["theta_hat"]

        if forcing is None and damping is None:
            # Use optimized path for no forcing/damping
            theta_hat_new = _rk4_step_optimized(
                theta_hat, self.grid, self.alpha, self.nu_p, float(self.p), dt
            )
        else:
            # Fall back to standard RK4 with forcing/damping
            def rhs_fn(theta_hat_):
                rhs = _compute_core_rhs_optimized(
                    theta_hat_, self.grid, self.alpha, self.nu_p, self.p
                )
                if forcing is not None:
                    rhs = rhs + forcing(theta_hat_, **kwargs)
                if damping is not None:
                    rhs = rhs + damping(theta_hat_, **kwargs)
                return rhs

            theta_hat_new = rk4_step(theta_hat, rhs_fn, dt)

        return {"theta_hat": theta_hat_new, "time": state["time"] + dt, "step": state["step"] + 1}

    def multistep(self, state: State, n_steps: int, dt: float) -> State:
        """
        Advance solution by multiple time steps using lax.scan.

        This is significantly faster than calling step() in a loop.
        Only works without forcing/damping.

        Parameters:
            state: Current state
            n_steps: Number of steps to take
            dt: Time step size

        Returns:
            Updated state after n_steps
        """
        theta_hat = state["theta_hat"]
        time = state["time"]

        # Use lax.scan for efficient multi-step integration
        scan_fn = partial(
            _multistep_scan,
            grid=self.grid,
            alpha=self.alpha,
            nu_p=self.nu_p,
            p=float(self.p),
            dt=dt,
        )
        (theta_hat_final, time_final), _ = lax.scan(
            scan_fn, (theta_hat, time), None, length=n_steps
        )

        return {"theta_hat": theta_hat_final, "time": time_final, "step": state["step"] + n_steps}

    def compute_velocity(self, theta_hat: jax.Array) -> tuple[Any, Any]:
        """Compute velocity field from θ."""
        return compute_velocity_from_theta(theta_hat, self.grid, self.alpha)

    def get_diagnostics(self, state: State) -> dict[str, float]:
        """Compute diagnostic quantities (same as base solver)."""
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

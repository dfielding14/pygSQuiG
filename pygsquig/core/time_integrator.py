"""
Time integration schemes for spectral simulations.

This module provides explicit time integrators optimized for
use with spectral methods in fluid dynamics simulations.
"""

from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp


def rk4_step(
    state: jax.Array, rhs_fn: Callable[..., jax.Array], dt: float, *args: Any
) -> jax.Array:
    """
    Fourth-order Runge-Kutta time step.

    Parameters:
        state: Current state (typically theta_hat)
        rhs_fn: Right-hand side function that takes (state, *args) -> time derivative
        dt: Time step size
        *args: Additional arguments passed to rhs_fn

    Returns:
        Updated state after one RK4 step
    """
    # RK4 stages
    k1 = rhs_fn(state, *args)
    k2 = rhs_fn(state + 0.5 * dt * k1, *args)
    k3 = rhs_fn(state + 0.5 * dt * k2, *args)
    k4 = rhs_fn(state + dt * k3, *args)

    # Combine stages
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def ssp_rk3_step(
    state: jax.Array, rhs_fn: Callable[..., jax.Array], dt: float, *args: Any
) -> jax.Array:
    """
    Strong Stability Preserving third-order Runge-Kutta time step.

    This method is useful for problems with discontinuities or
    when TVD (Total Variation Diminishing) properties are desired.

    Parameters:
        state: Current state (typically theta_hat)
        rhs_fn: Right-hand side function that takes (state, *args) -> time derivative
        dt: Time step size
        *args: Additional arguments passed to rhs_fn

    Returns:
        Updated state after one SSP-RK3 step
    """
    # Stage 1
    k1 = rhs_fn(state, *args)
    u1 = state + dt * k1

    # Stage 2
    k2 = rhs_fn(u1, *args)
    u2 = (3 / 4) * state + (1 / 4) * u1 + (1 / 4) * dt * k2

    # Stage 3
    k3 = rhs_fn(u2, *args)
    return (1 / 3) * state + (2 / 3) * u2 + (2 / 3) * dt * k3


def compute_cfl(
    u: jax.Array, v: jax.Array, dx: float, dy: float, safety_factor: float = 0.8
) -> float:
    """
    Compute CFL-limited time step for advection.

    For spectral methods, the effective grid spacing should account
    for the higher resolution of derivatives.

    Parameters:
        u: x-velocity field, shape (N, N)
        v: y-velocity field, shape (N, N)
        dx: Grid spacing in x
        dy: Grid spacing in y
        safety_factor: Safety factor < 1 (default 0.8)

    Returns:
        Maximum stable time step
    """
    # Maximum velocities
    u_max = jnp.abs(u).max()
    v_max = jnp.abs(v).max()

    # CFL condition: dt < min(dx/u_max, dy/v_max)
    # For spectral methods, use more conservative estimate
    dt_x = dx / (jnp.pi * u_max) if u_max > 0 else jnp.inf
    dt_y = dy / (jnp.pi * v_max) if v_max > 0 else jnp.inf

    dt_cfl = safety_factor * jnp.minimum(dt_x, dt_y)

    return dt_cfl


def compute_diffusion_timestep(nu: float, k2_max: float, safety_factor: float = 0.8) -> float:
    """
    Compute stable time step for diffusion terms.

    For explicit time stepping of diffusion terms ∂_t θ = ν Δ θ,
    stability requires dt < 2 / (ν k²_max).

    Parameters:
        nu: Diffusion coefficient
        k2_max: Maximum value of k² in the domain
        safety_factor: Safety factor < 1 (default 0.8)

    Returns:
        Maximum stable time step for diffusion
    """
    if nu <= 0 or k2_max <= 0:
        return jnp.inf

    dt_diff = 2.0 / (nu * k2_max)
    return safety_factor * dt_diff


def adaptive_timestep(
    u: jax.Array,
    v: jax.Array,
    grid_dx: float,
    nu: float = 0.0,
    k2_max: float = 0.0,
    dt_max: float = 0.1,
    cfl_safety: float = 0.8,
) -> float:
    """
    Compute adaptive time step based on CFL and diffusion constraints.

    Parameters:
        u: x-velocity field
        v: y-velocity field
        grid_dx: Grid spacing (assuming square grid)
        nu: Diffusion coefficient (optional)
        k2_max: Maximum k² for diffusion stability (optional)
        dt_max: Maximum allowed time step
        cfl_safety: CFL safety factor

    Returns:
        Recommended time step
    """
    # CFL constraint
    dt_cfl = compute_cfl(u, v, grid_dx, grid_dx, cfl_safety)

    # Diffusion constraint (if applicable)
    dt_diff = compute_diffusion_timestep(nu, k2_max, cfl_safety)

    # Take minimum of all constraints
    dt = jnp.minimum(dt_cfl, dt_diff)
    if dt_max is not None:
        dt = jnp.minimum(dt, dt_max)

    return dt

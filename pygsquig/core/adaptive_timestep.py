"""
Adaptive timestepping with CFL control for pygSQuiG.

This module provides adaptive timestep selection based on the
Courant-Friedrichs-Lewy (CFL) condition and other stability criteria.
"""

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array

from pygsquig.core.grid import Grid


@dataclass
class CFLConfig:
    """Configuration for CFL-based adaptive timestepping.

    Attributes:
        cfl_safety: Safety factor for CFL condition (0 < safety < 1)
        dt_min: Minimum allowed timestep
        dt_max: Maximum allowed timestep
        growth_factor: Maximum growth factor per step
        shrink_factor: Shrink factor when reducing timestep
        target_cfl: Target CFL number to maintain
        advection_weight: Weight for advection CFL (vs diffusion)
    """

    cfl_safety: float = 0.8
    dt_min: float = 1e-8
    dt_max: float = 1.0
    growth_factor: float = 1.1
    shrink_factor: float = 0.5
    target_cfl: float = 0.5
    advection_weight: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.cfl_safety <= 1:
            raise ValueError(f"cfl_safety must be in (0, 1], got {self.cfl_safety}")
        if not 0 < self.target_cfl <= 1:
            raise ValueError(f"target_cfl must be in (0, 1], got {self.target_cfl}")
        if self.dt_min >= self.dt_max:
            raise ValueError(f"dt_min ({self.dt_min}) must be less than dt_max ({self.dt_max})")
        if self.growth_factor <= 1:
            raise ValueError(f"growth_factor must be > 1, got {self.growth_factor}")
        if not 0 < self.shrink_factor < 1:
            raise ValueError(f"shrink_factor must be in (0, 1), got {self.shrink_factor}")


@jax.jit
def compute_max_velocity(u: Array, v: Array) -> float:
    """Compute maximum velocity magnitude.

    Args:
        u: x-velocity component
        v: y-velocity component

    Returns:
        Maximum velocity magnitude
    """
    vel_mag = jnp.sqrt(u**2 + v**2)
    return float(jnp.max(vel_mag))


@jax.jit
def compute_advection_cfl(u: Array, v: Array, dx: float) -> float:
    """Compute CFL number for advection.

    CFL_adv = max(|u|) * dt / dx

    Args:
        u: x-velocity component
        v: y-velocity component
        dx: Grid spacing

    Returns:
        Maximum timestep for advection CFL = 1
    """
    max_vel = compute_max_velocity(u, v)
    # Avoid division by zero
    max_vel = jnp.maximum(max_vel, 1e-10)
    return float(dx / max_vel)


def compute_diffusion_cfl(grid: Grid, nu: float, p: int = 2) -> float:
    """Compute CFL number for diffusion/hyperviscosity.

    For diffusion: CFL_diff = nu * dt / dx^2
    For hyperviscosity: CFL_hyp = nu * dt / dx^(2p)

    Args:
        grid: Grid object
        nu: Diffusivity/hyperviscosity coefficient
        p: Order of hyperviscosity

    Returns:
        Maximum timestep for diffusion CFL = 1
    """
    if nu == 0:
        return float(jnp.inf)

    dx = grid.L / grid.N

    # For spectral methods, the effective resolution is higher
    # due to the global nature of derivatives
    k_max = jnp.pi * grid.N / grid.L

    # Diffusion timestep constraint
    # dt < C * dx^(2p) / (nu * k_max^(2p-2))
    # For standard diffusion (p=1): dt < dx^2 / (4*nu)
    # For hyperviscosity (p>1): more restrictive

    if p == 1:
        # Standard diffusion
        dt_diff = 0.25 * dx**2 / nu
    else:
        # Hyperviscosity
        # The most restrictive mode is at k_max
        dt_diff = dx ** (2 * p) / (nu * k_max ** (2 * p - 2))

    return float(dt_diff)


def compute_timestep(
    u: Array,
    v: Array,
    grid: Grid,
    config: CFLConfig,
    nu_p: float = 0.0,
    p: int = 8,
    current_dt: Optional[float] = None,
) -> tuple[float, dict[str, float]]:
    """Compute adaptive timestep based on CFL conditions.

    Args:
        u: x-velocity in physical space
        v: y-velocity in physical space
        grid: Grid object
        config: CFL configuration
        nu_p: Hyperviscosity coefficient
        p: Hyperviscosity order
        current_dt: Current timestep (for growth limiting)

    Returns:
        Tuple of (new_dt, diagnostics)
    """
    dx = grid.L / grid.N

    # Advection CFL
    dt_adv = compute_advection_cfl(u, v, dx)

    # Diffusion CFL
    dt_diff = compute_diffusion_cfl(grid, nu_p, p)

    # Take the minimum of advection and diffusion constraints
    dt_combined = jnp.minimum(dt_adv, dt_diff)

    # Apply safety factor
    dt_safe = config.cfl_safety * dt_combined

    # Apply growth/shrink limits if we have current dt
    if current_dt is not None:
        # Limit growth
        dt_safe = jnp.minimum(dt_safe, current_dt * config.growth_factor)

        # Check if we need to shrink
        current_cfl_adv = current_dt / dt_adv
        if current_cfl_adv > 1.0:
            # CFL violation - shrink immediately
            dt_safe = current_dt * config.shrink_factor

    # Apply bounds
    dt_new = jnp.clip(dt_safe, config.dt_min, config.dt_max)

    # Compute diagnostics
    cfl_adv = dt_new / dt_adv
    cfl_diff = dt_new / dt_diff if dt_diff < jnp.inf else 0.0

    diagnostics = {
        "dt": float(dt_new),
        "dt_adv": float(dt_adv),
        "dt_diff": float(dt_diff),
        "cfl_adv": float(cfl_adv),
        "cfl_diff": float(cfl_diff),
        "max_velocity": float(compute_max_velocity(u, v)),
        "limited_by": "advection" if dt_adv < dt_diff else "diffusion",
    }

    return dt_new, diagnostics


class AdaptiveTimestepper:
    """Adaptive timestepping controller for pygSQuiG solver.

    This class manages adaptive timestep selection based on stability
    criteria and provides monitoring capabilities.
    """

    def __init__(self, grid: Grid, config: Optional[CFLConfig] = None, verbose: bool = False):
        """Initialize adaptive timestepper.

        Args:
            grid: Grid object
            config: CFL configuration (uses defaults if None)
            verbose: Print timestep changes
        """
        self.grid = grid
        self.config = config or CFLConfig()
        self.verbose = verbose

        # History tracking
        self.dt_history: list[float] = []
        self.cfl_history: list[float] = []
        self.time_history: list[float] = []

        # Statistics
        self.n_steps = 0
        self.n_rejected = 0
        self.n_dt_changes = 0

    def compute_timestep(
        self, state: dict[str, Any], u: Array, v: Array, nu_p: float = 0.0, p: int = 8
    ) -> tuple[float, dict[str, float]]:
        """Compute timestep for current state.

        Args:
            state: Current solver state
            u: x-velocity
            v: y-velocity
            nu_p: Hyperviscosity coefficient
            p: Hyperviscosity order

        Returns:
            Tuple of (dt, diagnostics)
        """
        # Get current dt if available
        current_dt = self.dt_history[-1] if self.dt_history else None

        # Compute new timestep
        dt_new, diags = compute_timestep(u, v, self.grid, self.config, nu_p, p, current_dt)

        # Check for significant changes
        if current_dt is not None:
            change_ratio = dt_new / current_dt
            if abs(change_ratio - 1.0) > 0.1:
                self.n_dt_changes += 1
                if self.verbose:
                    print(
                        f"Timestep change: {current_dt:.3e} -> {dt_new:.3e} "
                        f"(ratio {change_ratio:.2f})"
                    )

        # Update history
        self.dt_history.append(dt_new)
        self.cfl_history.append(diags["cfl_adv"])
        self.time_history.append(state.get("time", 0.0))
        self.n_steps += 1

        return dt_new, diags

    def check_stability(
        self, state_before: dict[str, Any], state_after: dict[str, Any], dt_used: float
    ) -> tuple[bool, Optional[str]]:
        """Check if timestep resulted in stable integration.

        Args:
            state_before: State before timestep
            state_after: State after timestep
            dt_used: Timestep that was used

        Returns:
            Tuple of (is_stable, reason_if_unstable)
        """
        theta_after = state_after["theta_hat"]

        # Check for NaN/Inf
        if jnp.any(jnp.isnan(theta_after)) or jnp.any(jnp.isinf(theta_after)):
            return False, "NaN or Inf detected"

        # Check for extreme growth
        theta_before = state_before["theta_hat"]
        growth = jnp.max(jnp.abs(theta_after)) / (jnp.max(jnp.abs(theta_before)) + 1e-10)

        if growth > 10.0:  # Arbitrary threshold
            return False, f"Excessive growth: {growth:.2f}x"

        # Check energy growth (for conservative systems)
        # This is problem-specific and might need adjustment

        return True, None

    def suggest_retry_timestep(self, dt_failed: float) -> float:
        """Suggest new timestep after failure.

        Args:
            dt_failed: Timestep that failed

        Returns:
            Suggested smaller timestep
        """
        self.n_rejected += 1
        dt_retry = dt_failed * self.config.shrink_factor

        if self.verbose:
            print(f"Timestep rejected, retrying with dt={dt_retry:.3e}")

        return max(dt_retry, self.config.dt_min)

    def get_statistics(self) -> dict[str, Any]:
        """Get timestepping statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.dt_history:
            return {}

        dt_array = jnp.array(self.dt_history)
        cfl_array = jnp.array(self.cfl_history)

        return {
            "n_steps": self.n_steps,
            "n_rejected": self.n_rejected,
            "n_dt_changes": self.n_dt_changes,
            "dt_min_used": float(jnp.min(dt_array)),
            "dt_max_used": float(jnp.max(dt_array)),
            "dt_mean": float(jnp.mean(dt_array)),
            "cfl_mean": float(jnp.mean(cfl_array)),
            "cfl_max": float(jnp.max(cfl_array)),
            "efficiency": 1.0 - self.n_rejected / (self.n_steps + self.n_rejected),
        }

    def reset_history(self):
        """Reset history and statistics."""
        self.dt_history.clear()
        self.cfl_history.clear()
        self.time_history.clear()
        self.n_steps = 0
        self.n_rejected = 0
        self.n_dt_changes = 0


def estimate_initial_timestep(
    grid: Grid, nu_p: float = 0.0, p: int = 8, target_cfl: float = 0.5
) -> float:
    """Estimate reasonable initial timestep.

    Args:
        grid: Grid object
        nu_p: Hyperviscosity coefficient
        p: Hyperviscosity order
        target_cfl: Target CFL number

    Returns:
        Estimated initial timestep
    """
    dx = grid.L / grid.N

    # Assume moderate initial velocities
    # For gSQG, typical velocities scale with system size
    u_typical = 1.0  # Adjust based on problem

    # Advection constraint
    dt_adv = target_cfl * dx / u_typical

    # Diffusion constraint if present
    if nu_p > 0:
        dt_diff = compute_diffusion_cfl(grid, nu_p, p)
        dt_est = min(dt_adv, dt_diff * target_cfl)
    else:
        dt_est = dt_adv

    return dt_est

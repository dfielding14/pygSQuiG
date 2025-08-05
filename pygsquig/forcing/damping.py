"""
Damping mechanisms for turbulence simulations.

This module implements large-scale damping to prevent energy accumulation
at scales larger than the forcing scale.
"""

import jax
import jax.numpy as jnp

from pygsquig.core.grid import Grid
from pygsquig.core.operators import hyperviscosity


def large_scale_damping(theta_hat: jax.Array, grid: Grid, mu: float, kf: float) -> jax.Array:
    """
    Apply linear damping to large-scale modes.

    Implements -μθ for modes with |k| < kf/2, which prevents
    energy accumulation at scales larger than the forcing scale.

    Parameters:
        theta_hat: Fourier coefficients of scalar field
        grid: Grid object
        mu: Damping coefficient
        kf: Forcing wavenumber (damping applied for k < kf/2)

    Returns:
        Damping term in spectral space
    """
    if mu == 0:
        return jnp.zeros_like(theta_hat)

    # Compute wavenumber magnitude
    k_mag = jnp.sqrt(grid.k2)

    # Create mask for large scales (k < kf/2)
    mask = k_mag < kf / 2

    # Apply damping: -μθ for large scales
    damping = -mu * theta_hat * mask

    return damping


# Note: hyperviscosity function is now imported from operators module


class CombinedDamping:
    """
    Combined damping with both large-scale and small-scale dissipation.

    This class provides a convenient interface for applying both
    large-scale damping and hyperviscosity together.
    """

    def __init__(self, mu: float = 0.0, kf: float = 20.0, nu_p: float = 0.0, p: int = 8):
        """
        Initialize combined damping.

        Parameters:
            mu: Large-scale damping coefficient
            kf: Forcing wavenumber (for large-scale cutoff)
            nu_p: Hyperviscosity coefficient
            p: Hyperviscosity order
        """
        self.mu = mu
        self.kf = kf
        self.nu_p = nu_p
        self.p = p

        if p not in [2, 4, 8]:
            raise ValueError(f"p must be 2, 4, or 8, got {p}")

    def __call__(self, theta_hat: jax.Array, grid: Grid) -> jax.Array:
        """
        Apply combined damping.

        Parameters:
            theta_hat: Current state in spectral space
            grid: Grid object

        Returns:
            Total damping term
        """
        # Large-scale damping
        large_scale = large_scale_damping(theta_hat, grid, self.mu, self.kf)

        # Small-scale hyperviscosity
        small_scale = hyperviscosity(theta_hat, grid, self.nu_p, self.p)

        return large_scale + small_scale

    def get_diagnostics(self, theta_hat: jax.Array, grid: Grid) -> dict:
        """
        Compute damping diagnostics.

        Parameters:
            theta_hat: Current state
            grid: Grid object

        Returns:
            Dictionary with diagnostic values
        """
        # Compute individual contributions
        large_scale = large_scale_damping(theta_hat, grid, self.mu, self.kf)
        small_scale = hyperviscosity(theta_hat, grid, self.nu_p, self.p)

        # Energy dissipation rates
        from pygsquig.core.grid import ifft2

        theta = ifft2(theta_hat)

        large_scale_phys = ifft2(large_scale)
        small_scale_phys = ifft2(small_scale)

        large_scale_dissipation = -jnp.mean(theta * large_scale_phys)
        small_scale_dissipation = -jnp.mean(theta * small_scale_phys)

        # Count affected modes
        k_mag = jnp.sqrt(grid.k2)
        n_large_scale_modes = jnp.sum(k_mag < self.kf / 2)

        return {
            "large_scale_dissipation": float(large_scale_dissipation),
            "small_scale_dissipation": float(small_scale_dissipation),
            "total_dissipation": float(large_scale_dissipation + small_scale_dissipation),
            "n_large_scale_modes": int(n_large_scale_modes),
            "mu": self.mu,
            "nu_p": self.nu_p,
        }

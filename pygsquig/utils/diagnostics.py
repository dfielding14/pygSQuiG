"""Diagnostic functions for pygSQuiG simulations.

This module provides functions to compute various diagnostic quantities
such as energy spectra, fluxes, and other relevant statistics.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial

from ..core.grid import Grid
from ..core.operators import (
    compute_velocity_from_theta,
    fractional_laplacian,
    jacobian,
)


@partial(jax.jit, static_argnums=(2,))
def _compute_energy_density(
    theta_hat: jnp.ndarray,
    k2: jnp.ndarray,
    alpha: float
) -> jnp.ndarray:
    """Compute energy density in Fourier space (JIT-compiled helper)."""
    k_mag = jnp.sqrt(k2)
    k_safe = jnp.where(k_mag > 0, k_mag, 1.0)
    energy_density = 0.5 * k_safe**(alpha - 2) * jnp.abs(theta_hat)**2
    energy_density = energy_density.at[0, 0].set(0.0)
    return energy_density, k_mag


def compute_energy_spectrum(
    theta_hat: jnp.ndarray,
    grid: Grid,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the energy spectrum E(k) for gSQG.
    
    For gSQG, the energy is defined as:
    E = (1/2) ⟨|(-Δ)^((α-2)/2) θ|²⟩
    
    The energy spectrum E(k) gives the energy in wavenumber shells.
    
    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent in velocity relation
        
    Returns:
        k_bins: Wavenumber bin centers
        E_k: Energy spectrum E(k)
    """
    # Compute energy density using JIT-compiled function
    energy_density, k_mag = _compute_energy_density(theta_hat, grid.k2, alpha)
    
    # Compute radial spectrum
    # Define wavenumber bins
    dk = 2 * np.pi / grid.L  # Fundamental wavenumber
    k_max = grid.N * dk / 2  # Maximum resolved wavenumber
    
    # Create logarithmically spaced bins for better resolution at all scales
    n_bins = min(grid.N // 4, 64)  # Reasonable number of bins
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
    
    # Compute spectrum by binning
    E_k = np.zeros(n_bins)
    
    # Convert to numpy for binning
    k_mag_np = np.array(k_mag)
    energy_density_np = np.array(energy_density)
    
    for i in range(n_bins):
        # Find modes in this bin
        mask = (k_mag_np >= k_edges[i]) & (k_mag_np < k_edges[i+1])
        
        # Sum energy in this bin
        # Note: multiply by 2 for negative frequencies (except k_x = 0)
        E_k[i] = np.sum(energy_density_np[mask])
        
        # Account for complex conjugate pairs
        # For rfft2 format, we would need different handling
    
    # Normalize by bin width for energy per unit wavenumber
    E_k /= (k_edges[1:] - k_edges[:-1])
    
    return k_bins, E_k


@jax.jit
def compute_scalar_flux(
    theta_hat: jnp.ndarray,
    velocity: Tuple[jnp.ndarray, jnp.ndarray],
    grid: Grid
) -> jnp.ndarray:
    """Compute the scalar flux ⟨u·∇θ²⟩.
    
    This quantity measures the cascade rate of scalar variance.
    
    Args:
        theta_hat: Fourier coefficients of theta
        velocity: Tuple of (u, v) velocity components in physical space
        grid: Grid object
        
    Returns:
        Scalar flux value
    """
    # Get theta in physical space
    from ..core.grid import ifft2
    theta = ifft2(theta_hat)
    u, v = velocity
    
    # Compute u·∇θ² = 2θ(u·∇θ)
    # First compute u·∇θ using the Jacobian
    u_dot_grad_theta = jacobian(theta, u, v, grid)
    
    # Compute flux
    flux = 2 * jnp.mean(theta * u_dot_grad_theta)
    
    return flux


@partial(jax.jit, static_argnums=(2,))
def compute_enstrophy(
    theta_hat: jnp.ndarray,
    grid: Grid,
    alpha: float
) -> jnp.ndarray:
    """Compute the enstrophy Ω = (1/2)⟨|(-Δ)^(α/2)θ|²⟩.
    
    For gSQG, the generalized vorticity is q = (-Δ)^(α/2)θ,
    and enstrophy is the mean square vorticity.
    
    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        
    Returns:
        Enstrophy value
    """
    # Compute generalized vorticity in Fourier space
    q_hat = fractional_laplacian(theta_hat, grid, alpha)
    
    # Enstrophy is mean square vorticity
    # In spectral space: Ω = (1/2) ∑_k |q_hat_k|²
    enstrophy = 0.5 * jnp.mean(jnp.abs(q_hat)**2) * (grid.N**2)
    
    return enstrophy


@jax.jit
def compute_energy_flux(
    theta_hat: jnp.ndarray,
    forcing_hat: Optional[jnp.ndarray],
    grid: Grid
) -> jnp.ndarray:
    """Compute the energy flux ⟨θF⟩.
    
    This measures the rate of energy injection by forcing.
    
    Args:
        theta_hat: Fourier coefficients of theta
        forcing_hat: Fourier coefficients of forcing (can be None)
        grid: Grid object
        
    Returns:
        Energy flux value
    """
    if forcing_hat is None:
        return 0.0
    
    # Energy injection rate is ⟨θF⟩
    # In spectral space: ∑_k Re(θ_hat_k* F_hat_k)
    flux = jnp.real(jnp.sum(jnp.conj(theta_hat) * forcing_hat)) / (grid.N**2)
    
    return flux


@partial(jax.jit, static_argnums=(2,))
def compute_total_energy(
    theta_hat: jnp.ndarray,
    grid: Grid,
    alpha: float
) -> jnp.ndarray:
    """Compute total energy E = (1/2)⟨|(-Δ)^((α-2)/2)θ|²⟩.
    
    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object  
        alpha: Fractional exponent
        
    Returns:
        Total energy
    """
    # In spectral space
    k2_safe = jnp.where(grid.k2 > 0, grid.k2, 1.0)
    energy_density = 0.5 * k2_safe**((alpha - 2) / 2) * jnp.abs(theta_hat)**2
    energy_density = energy_density.at[0, 0].set(0.0)  # No mean flow
    
    # Total energy
    energy = jnp.sum(energy_density) / (grid.N**2)
    
    return energy


@partial(jax.jit, static_argnums=(2,))
def compute_palinstrophy(
    theta_hat: jnp.ndarray,
    grid: Grid,
    alpha: float
) -> jnp.ndarray:
    """Compute palinstrophy P = (1/2)⟨|∇q|²⟩ where q = (-Δ)^(α/2)θ.
    
    This is the mean square gradient of vorticity.
    
    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        
    Returns:
        Palinstrophy value
    """
    # q = (-Δ)^(α/2)θ, so in Fourier space: q_hat = |k|^α θ_hat
    # ∇q has Fourier coefficients: ik q_hat
    # |∇q|² = |k|² |q_hat|² = |k|^(2+2α) |θ_hat|²
    
    k2_safe = jnp.where(grid.k2 > 0, grid.k2, 1.0)
    palinstrophy_density = 0.5 * k2_safe**(1 + alpha) * jnp.abs(theta_hat)**2
    palinstrophy_density = palinstrophy_density.at[0, 0].set(0.0)
    
    palinstrophy = jnp.sum(palinstrophy_density) / (grid.N**2)
    
    return palinstrophy
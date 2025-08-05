"""
Diagnostic functions for passive scalar analysis.

This module provides functions to compute various diagnostics
for passive scalar fields including spectra, fluxes, and mixing metrics.
"""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from pygsquig.core.grid import Grid, ifft2


def compute_scalar_variance_spectrum(
    scalar_hat: jnp.ndarray,
    grid: Grid
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute spectrum of scalar variance.
    
    For a passive scalar θ, this computes the variance spectrum
    V(k) such that ∫V(k)dk = ⟨θ²⟩.
    
    Args:
        scalar_hat: Scalar field in spectral space
        grid: Grid object
        
    Returns:
        Tuple of (k_bins, variance_spectrum)
    """
    # Compute scalar variance density in spectral space
    variance_density = 0.5 * jnp.abs(scalar_hat)**2
    
    # Remove mean (k=0 mode)
    variance_density = variance_density.at[0, 0].set(0.0)
    
    # Compute radial average
    k_mag = jnp.sqrt(grid.k2)
    
    # Define wavenumber bins
    dk = 2 * np.pi / grid.L
    k_max = grid.N * dk / 2
    n_bins = min(grid.N // 4, 64)
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
    
    # Bin the spectrum
    spectrum = np.zeros(n_bins)
    k_mag_np = np.array(k_mag)
    variance_density_np = np.array(variance_density)
    
    for i in range(n_bins):
        mask = (k_mag_np >= k_edges[i]) & (k_mag_np < k_edges[i+1])
        spectrum[i] = np.sum(variance_density_np[mask])
    
    # Normalize by bin width
    spectrum /= (k_edges[1:] - k_edges[:-1])
    
    # Normalize by domain size
    spectrum = spectrum * (grid.L / (2 * np.pi))**2
    
    return k_bins, spectrum


def compute_scalar_flux(
    scalar_hat: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: Grid
) -> Tuple[float, float]:
    """Compute turbulent scalar flux ⟨u'θ'⟩.
    
    This measures the correlation between velocity fluctuations
    and scalar fluctuations, indicating turbulent transport.
    
    Args:
        scalar_hat: Scalar field in spectral space
        u: x-velocity in physical space
        v: y-velocity in physical space
        grid: Grid object
        
    Returns:
        Tuple of (flux_x, flux_y) components
    """
    # Convert scalar to physical space
    scalar = ifft2(scalar_hat).real
    
    # Remove means to get fluctuations
    scalar_prime = scalar - jnp.mean(scalar)
    u_prime = u - jnp.mean(u)
    v_prime = v - jnp.mean(v)
    
    # Compute fluxes
    flux_x = float(jnp.mean(u_prime * scalar_prime))
    flux_y = float(jnp.mean(v_prime * scalar_prime))
    
    return flux_x, flux_y


def compute_scalar_dissipation(
    scalar_hat: jnp.ndarray,
    grid: Grid,
    kappa: float
) -> float:
    """Compute scalar dissipation rate χ = κ⟨|∇θ|²⟩.
    
    This measures the rate at which scalar variance is destroyed
    by molecular diffusion.
    
    Args:
        scalar_hat: Scalar field in spectral space
        grid: Grid object
        kappa: Diffusivity coefficient
        
    Returns:
        Scalar dissipation rate
    """
    if kappa == 0:
        return 0.0
        
    # Compute |∇θ|² in spectral space
    # |∇θ|² = |∂θ/∂x|² + |∂θ/∂y|² = k²|θ̂|²
    grad_squared_hat = grid.k2 * jnp.abs(scalar_hat)**2
    
    # Integrate over all wavenumbers with proper normalization
    # The mean of |∇θ|² in physical space equals the sum in k-space divided by N⁴
    # because FFT normalization gives |θ̂|² = N² * |θ|²_phys for each mode
    chi = kappa * jnp.sum(grad_squared_hat).real / (grid.N**4)
    
    return float(chi)


def compute_mixing_efficiency(
    scalar_hat_initial: jnp.ndarray,
    scalar_hat_final: jnp.ndarray,
    grid: Grid,
    time_elapsed: float
) -> float:
    """Compute mixing efficiency based on variance decay.
    
    Mixing efficiency is defined as the rate of variance decay
    normalized by the initial variance.
    
    Args:
        scalar_hat_initial: Initial scalar field
        scalar_hat_final: Final scalar field
        grid: Grid object
        time_elapsed: Time between initial and final states
        
    Returns:
        Mixing efficiency (dimensionless)
    """
    # Compute variances
    var_initial = float(compute_scalar_variance(scalar_hat_initial))
    var_final = float(compute_scalar_variance(scalar_hat_final))
    
    # Mixing efficiency
    if var_initial > 0 and time_elapsed > 0:
        efficiency = (var_initial - var_final) / (var_initial * time_elapsed)
    else:
        efficiency = 0.0
        
    return float(efficiency)


@jax.jit
def compute_scalar_variance(scalar_hat: jnp.ndarray) -> jnp.ndarray:
    """Compute total scalar variance ⟨θ²⟩ (JIT-compiled).
    
    Args:
        scalar_hat: Scalar field in spectral space
        
    Returns:
        Total variance (as JAX array for JIT compatibility)
    """
    scalar = ifft2(scalar_hat).real
    # Remove mean before computing variance
    scalar_prime = scalar - jnp.mean(scalar)
    return jnp.mean(scalar_prime**2)


def compute_scalar_gradient_alignment(
    scalar_hat: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: Grid
) -> float:
    """Compute alignment between scalar gradient and strain.
    
    This diagnostic measures how well the scalar gradient aligns
    with the compressive direction of the strain rate tensor,
    which enhances mixing.
    
    Args:
        scalar_hat: Scalar field in spectral space
        u: x-velocity in physical space
        v: y-velocity in physical space
        grid: Grid object
        
    Returns:
        Average alignment cosine
    """
    # Compute scalar gradient in physical space
    dtheta_dx_hat = 1j * grid.kx * scalar_hat
    dtheta_dy_hat = 1j * grid.ky * scalar_hat
    dtheta_dx = ifft2(dtheta_dx_hat).real
    dtheta_dy = ifft2(dtheta_dy_hat).real
    
    # Compute velocity gradients
    u_hat = fft2(u)
    v_hat = fft2(v)
    du_dx = ifft2(1j * grid.kx * u_hat).real
    du_dy = ifft2(1j * grid.ky * u_hat).real
    dv_dx = ifft2(1j * grid.kx * v_hat).real
    dv_dy = ifft2(1j * grid.ky * v_hat).real
    
    # Strain rate tensor components
    S11 = du_dx
    S22 = dv_dy
    S12 = 0.5 * (du_dy + dv_dx)
    
    # Compute alignment (simplified metric)
    # Full implementation would compute eigenvalues/eigenvectors
    grad_mag_sq = dtheta_dx**2 + dtheta_dy**2
    strain_mag_sq = S11**2 + S22**2 + 2*S12**2
    
    # Avoid division by zero
    mask = (grad_mag_sq > 1e-10) & (strain_mag_sq > 1e-10)
    
    # Simplified alignment metric
    alignment = jnp.where(
        mask,
        (dtheta_dx * S11 * dtheta_dx + 
         dtheta_dy * S22 * dtheta_dy + 
         2 * dtheta_dx * S12 * dtheta_dy) / 
        (jnp.sqrt(grad_mag_sq) * jnp.sqrt(strain_mag_sq)),
        0.0
    )
    
    return float(jnp.mean(alignment))


def compute_scalar_pdf_moments(
    scalar_hat: jnp.ndarray,
    max_moment: int = 4
) -> dict:
    """Compute statistical moments of scalar PDF.
    
    Args:
        scalar_hat: Scalar field in spectral space
        max_moment: Maximum moment to compute
        
    Returns:
        Dictionary with moments (mean, variance, skewness, kurtosis, etc.)
    """
    scalar = ifft2(scalar_hat).real
    
    moments = {}
    moments['mean'] = float(jnp.mean(scalar))
    
    # Central moments
    scalar_centered = scalar - moments['mean']
    
    for n in range(2, max_moment + 1):
        moments[f'moment_{n}'] = float(jnp.mean(scalar_centered**n))
    
    # Normalized moments
    if moments['moment_2'] > 0:
        std = jnp.sqrt(moments['moment_2'])
        moments['skewness'] = moments['moment_3'] / std**3
        moments['kurtosis'] = moments['moment_4'] / std**4 - 3.0
    else:
        moments['skewness'] = 0.0
        moments['kurtosis'] = 0.0
        
    return moments


def compute_batchelor_scale(
    scalar_hat: jnp.ndarray,
    grid: Grid,
    kappa: float,
    epsilon: float
) -> float:
    """Compute Batchelor scale for passive scalar.
    
    The Batchelor scale l_B = (κ³/ε)^(1/4) is the scale at which
    scalar gradients are smoothed by molecular diffusion.
    
    Args:
        scalar_hat: Scalar field in spectral space
        grid: Grid object
        kappa: Scalar diffusivity
        epsilon: Turbulent dissipation rate
        
    Returns:
        Batchelor scale
    """
    if kappa == 0 or epsilon == 0:
        return np.inf
        
    l_B = (kappa**3 / epsilon)**(1/4)
    return float(l_B)


def compute_peclet_number(
    u_rms: float,
    L_integral: float,
    kappa: float
) -> float:
    """Compute Péclet number Pe = U*L/κ.
    
    The Péclet number measures the ratio of advective to
    diffusive transport.
    
    Args:
        u_rms: RMS velocity
        L_integral: Integral length scale
        kappa: Scalar diffusivity
        
    Returns:
        Péclet number
    """
    if kappa == 0:
        return np.inf
        
    Pe = u_rms * L_integral / kappa
    return float(Pe)


# Import fft2 for gradient alignment computation
from pygsquig.core.grid import fft2
"""
Grid management for 2D spectral simulations.

This module provides the Grid dataclass and associated functions for managing
spatial discretization and spectral operations in doubly-periodic domains.
"""

from typing import Tuple, Any
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node


class Grid:
    """
    Grid information for 2D spectral simulations.
    
    This class is registered as a JAX pytree to enable JIT compilation
    and other JAX transformations.
    
    Attributes:
        N: Number of points per dimension
        L: Domain size (assumed square, [0, L] × [0, L])
        x: Physical x-coordinates, shape (N, N)
        y: Physical y-coordinates, shape (N, N)
        kx: Wavenumbers in x-direction, shape (N, N)
        ky: Wavenumbers in y-direction, shape (N, N)
        k2: k² = kx² + ky², shape (N, N)
        dealias_mask: 2/3 dealiasing mask, shape (N, N)
    """
    
    def __init__(self, N: int, L: float, x: jax.Array, y: jax.Array,
                 kx: jax.Array, ky: jax.Array, k2: jax.Array,
                 dealias_mask: jax.Array):
        self.N = N
        self.L = L
        self.x = x
        self.y = y
        self.kx = kx
        self.ky = ky
        self.k2 = k2
        self.dealias_mask = dealias_mask
    
    def tree_flatten(self) -> Tuple[list, dict]:
        """Flatten Grid into JAX-compatible format."""
        # Arrays that should be traced by JAX
        children = [self.x, self.y, self.kx, self.ky, self.k2, self.dealias_mask]
        # Static data that doesn't change
        aux_data = {'N': self.N, 'L': self.L}
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> 'Grid':
        """Reconstruct Grid from flattened representation."""
        x, y, kx, ky, k2, dealias_mask = children
        return cls(aux_data['N'], aux_data['L'], x, y, kx, ky, k2, dealias_mask)


# Register Grid as a JAX pytree
register_pytree_node(
    Grid,
    Grid.tree_flatten,
    Grid.tree_unflatten
)


@partial(jax.jit, static_argnums=(0, 1))
def make_grid(N: int, L: float) -> Grid:
    """
    Create a Grid object for spectral simulations.
    
    Parameters:
        N: Number of points per dimension (must be even)
        L: Domain size
        
    Returns:
        Grid object with all necessary arrays
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even, got {N}")
    
    # Physical space coordinates
    dx = L / N
    x1d = jnp.arange(N) * dx
    x, y = jnp.meshgrid(x1d, x1d, indexing='ij')
    
    # Wavenumber arrays (properly ordered for FFT)
    # For even N: [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]
    k1d = jnp.fft.fftfreq(N, d=dx) * 2 * jnp.pi
    kx, ky = jnp.meshgrid(k1d, k1d, indexing='ij')
    
    # k² for Laplacian operations
    k2 = kx**2 + ky**2
    
    # 2/3 dealiasing mask
    # Keep modes with |k| < (2/3) * k_max, where k_max = π * N / L
    k_max = jnp.pi * N / L  # Maximum wavenumber
    dealias_cutoff = (2.0 / 3.0) * k_max
    k_mag = jnp.sqrt(k2)
    dealias_mask = k_mag < dealias_cutoff
    
    return Grid(N, L, x, y, kx, ky, k2, dealias_mask)


@jax.jit
def fft2(field: jax.Array) -> jax.Array:
    """
    2D Fast Fourier Transform.
    
    Note: JAX FFT convention preserves Parseval's theorem with factor 1/N²:
    ⟨|f|²⟩ = (1/N²) ⟨|f̂|²⟩
    
    Parameters:
        field: Real-space field, shape (N, N)
        
    Returns:
        Fourier coefficients, shape (N, N)
    """
    return jnp.fft.fft2(field)


@jax.jit
def ifft2(field_hat: jax.Array) -> jax.Array:
    """
    2D Inverse Fast Fourier Transform.
    
    Parameters:
        field_hat: Fourier coefficients, shape (N, N)
        
    Returns:
        Real-space field, shape (N, N)
    """
    return jnp.fft.ifft2(field_hat).real
"""
Spectral operators for 2D turbulence simulations.

This module implements differential operators in spectral space for
the generalized Surface Quasi-Geostrophic (gSQG) equations.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from pygsquig.core.grid import Grid, fft2, ifft2


@jax.jit
def gradient(theta_hat: jax.Array, grid: Grid) -> Tuple[jax.Array, jax.Array]:
    """
    Compute gradient of a scalar field in physical space.

    Parameters:
        theta_hat: Fourier coefficients of scalar field, shape (N, N)
        grid: Grid object

    Returns:
        (dtheta_dx, dtheta_dy): Gradient components in physical space
    """
    # Spectral derivatives: ∂x → ikx, ∂y → iky
    dtheta_dx_hat = 1j * grid.kx * theta_hat
    dtheta_dy_hat = 1j * grid.ky * theta_hat

    # Transform to physical space
    dtheta_dx = ifft2(dtheta_dx_hat)
    dtheta_dy = ifft2(dtheta_dy_hat)

    return dtheta_dx, dtheta_dy


@jax.jit
def laplacian(theta_hat: jax.Array, grid: Grid) -> jax.Array:
    """
    Apply spectral Laplacian operator.

    Parameters:
        theta_hat: Fourier coefficients, shape (N, N)
        grid: Grid object

    Returns:
        Laplacian of theta in spectral space
    """
    return -grid.k2 * theta_hat


@jax.jit
def fractional_laplacian(theta_hat: jax.Array, grid: Grid, alpha: float) -> jax.Array:
    """
    Apply fractional Laplacian (-Δ)^(α/2).

    Parameters:
        theta_hat: Fourier coefficients, shape (N, N)
        grid: Grid object
        alpha: Fractional power (typically between -2 and 2)

    Returns:
        (-Δ)^(α/2) theta in spectral space
    """
    # (-Δ)^(α/2) → |k|^α in Fourier space
    k_mag = jnp.sqrt(grid.k2)

    # Handle zero mode separately to avoid division by zero
    k_mag_safe = jnp.where(grid.k2 > 0, k_mag, 1.0)
    fractional_op = jnp.where(grid.k2 > 0, k_mag_safe**alpha, 0.0)

    return fractional_op * theta_hat


@jax.jit
def perpendicular_gradient(psi_hat: jax.Array, grid: Grid) -> Tuple[jax.Array, jax.Array]:
    """
    Compute perpendicular gradient ∇^⊥ψ = (-∂yψ, ∂xψ).

    This gives the velocity field from a streamfunction.

    Parameters:
        psi_hat: Fourier coefficients of streamfunction, shape (N, N)
        grid: Grid object

    Returns:
        (u, v): Velocity components in physical space
    """
    # ∇^⊥ψ = (-∂yψ, ∂xψ)
    u_hat = -1j * grid.ky * psi_hat
    v_hat = 1j * grid.kx * psi_hat

    # Transform to physical space
    u = ifft2(u_hat)
    v = ifft2(v_hat)

    return u, v


@jax.jit
def jacobian(theta: jax.Array, u: jax.Array, v: jax.Array, grid: Grid) -> jax.Array:
    """
    Compute the Jacobian J(θ, ψ) = u·∇θ with dealiasing.

    This represents advection of θ by velocity field (u, v).

    Parameters:
        theta: Scalar field in physical space, shape (N, N)
        u: x-velocity in physical space, shape (N, N)
        v: y-velocity in physical space, shape (N, N)
        grid: Grid object

    Returns:
        J(θ, ψ) in physical space
    """
    # Transform theta to spectral space for derivatives
    theta_hat = fft2(theta)

    # Compute gradients of theta
    dtheta_dx, dtheta_dy = gradient(theta_hat, grid)

    # Compute advection term in physical space
    jacobian_phys = u * dtheta_dx + v * dtheta_dy

    # Transform to spectral space and apply dealiasing
    jacobian_hat = fft2(jacobian_phys)
    jacobian_hat = jacobian_hat * grid.dealias_mask

    # Return in physical space
    return ifft2(jacobian_hat)


@jax.jit
def compute_streamfunction(theta_hat: jax.Array, grid: Grid, alpha: float) -> jax.Array:
    """
    Compute streamfunction from vorticity using (-Δ)^(-α/2).

    For gSQG: ψ = (-Δ)^(-α/2)θ

    Parameters:
        theta_hat: Fourier coefficients of scalar field, shape (N, N)
        grid: Grid object
        alpha: Fractional power for gSQG dynamics

    Returns:
        psi_hat: Fourier coefficients of streamfunction
    """
    # ψ = (-Δ)^(-α/2)θ → |k|^(-α) in Fourier space
    k_mag = jnp.sqrt(grid.k2)

    # Handle zero mode
    k_mag_safe = jnp.where(grid.k2 > 0, k_mag, 1.0)
    inv_fractional_op = jnp.where(grid.k2 > 0, k_mag_safe ** (-alpha), 0.0)

    return inv_fractional_op * theta_hat


@jax.jit
def compute_velocity_from_theta(
    theta_hat: jax.Array, grid: Grid, alpha: float
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute velocity field from scalar field θ for gSQG dynamics.

    u = ∇^⊥(-Δ)^(-α/2)θ

    Parameters:
        theta_hat: Fourier coefficients of scalar field, shape (N, N)
        grid: Grid object
        alpha: Fractional power for gSQG dynamics

    Returns:
        (u, v): Velocity components in physical space
    """
    # First compute streamfunction
    psi_hat = compute_streamfunction(theta_hat, grid, alpha)

    # Then get velocity from perpendicular gradient
    u, v = perpendicular_gradient(psi_hat, grid)

    return u, v


@jax.jit
def hyperviscosity(theta_hat: jax.Array, grid: Grid, nu_p: float, p: int) -> jax.Array:
    """
    Apply hyperviscosity dissipation operator.

    The hyperviscosity term is: -ν_p (-Δ)^p θ
    In spectral space: -ν_p k^(2p) θ̂

    Parameters:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        nu_p: Hyperviscosity coefficient
        p: Order of hyperviscosity (even integer, typically 2, 4, or 8)

    Returns:
        Hyperviscosity term in spectral space
    """
    # Dissipation: -ν_p k^(2p) θ̂
    dissipation = -nu_p * (grid.k2 ** (p / 2)) * theta_hat

    # Ensure no dissipation of mean (k=0 mode)
    dissipation = dissipation.at[0, 0].set(0.0)

    return dissipation

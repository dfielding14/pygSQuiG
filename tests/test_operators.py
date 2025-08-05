"""
Tests for spectral operators module.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pygsquig.core.grid import make_grid, fft2, ifft2
from pygsquig.core.operators import (
    gradient,
    laplacian,
    fractional_laplacian,
    perpendicular_gradient,
    jacobian,
    compute_streamfunction,
    compute_velocity_from_theta,
)


class TestGradient:
    """Tests for gradient operator."""

    def test_gradient_sinusoidal(self):
        """Test gradient on sinusoidal functions."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Test function: sin(2x)
        theta = jnp.sin(2 * grid.x)
        theta_hat = fft2(theta)

        dtheta_dx, dtheta_dy = gradient(theta_hat, grid)

        # Expected: ∂x sin(2x) = 2cos(2x), ∂y sin(2x) = 0
        expected_dx = 2 * jnp.cos(2 * grid.x)
        expected_dy = jnp.zeros_like(grid.x)

        np.testing.assert_allclose(dtheta_dx, expected_dx, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dtheta_dy, expected_dy, rtol=1e-8, atol=1e-10)

    def test_gradient_2d_function(self):
        """Test gradient on 2D function."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Test function: sin(x) * cos(2y)
        theta = jnp.sin(grid.x) * jnp.cos(2 * grid.y)
        theta_hat = fft2(theta)

        dtheta_dx, dtheta_dy = gradient(theta_hat, grid)

        # Expected gradients
        expected_dx = jnp.cos(grid.x) * jnp.cos(2 * grid.y)
        expected_dy = -2 * jnp.sin(grid.x) * jnp.sin(2 * grid.y)

        np.testing.assert_allclose(dtheta_dx, expected_dx, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dtheta_dy, expected_dy, rtol=1e-8, atol=1e-10)

    def test_gradient_constant(self):
        """Test gradient of constant field is zero."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        theta = jnp.ones((N, N))
        theta_hat = fft2(theta)

        dtheta_dx, dtheta_dy = gradient(theta_hat, grid)

        np.testing.assert_allclose(dtheta_dx, 0, atol=1e-10)
        np.testing.assert_allclose(dtheta_dy, 0, atol=1e-10)


class TestLaplacian:
    """Tests for Laplacian operator."""

    def test_laplacian_eigenfunction(self):
        """Test Laplacian on its eigenfunctions."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Eigenfunction: sin(kx*x) * sin(ky*y)
        kx_mode, ky_mode = 3, 2
        theta = jnp.sin(kx_mode * grid.x) * jnp.sin(ky_mode * grid.y)
        theta_hat = fft2(theta)

        lap_theta_hat = laplacian(theta_hat, grid)
        lap_theta = ifft2(lap_theta_hat)

        # Expected: -(kx² + ky²) * theta
        eigenvalue = -(kx_mode**2 + ky_mode**2)
        expected = eigenvalue * theta

        np.testing.assert_allclose(lap_theta, expected, rtol=1e-8, atol=1e-10)

    def test_laplacian_gaussian(self):
        """Test Laplacian on Gaussian."""
        N = 256
        L = 4 * np.pi
        grid = make_grid(N=N, L=L)

        # Gaussian: exp(-r²/2σ²)
        sigma = L / 8
        r2 = (grid.x - L / 2) ** 2 + (grid.y - L / 2) ** 2
        theta = jnp.exp(-r2 / (2 * sigma**2))
        theta_hat = fft2(theta)

        lap_theta_hat = laplacian(theta_hat, grid)
        lap_theta = ifft2(lap_theta_hat)

        # Analytical Laplacian of Gaussian
        expected = theta * (r2 / sigma**4 - 2 / sigma**2)

        # Check at center where value is largest
        center = N // 2
        np.testing.assert_allclose(
            lap_theta[center - 5 : center + 5, center - 5 : center + 5],
            expected[center - 5 : center + 5, center - 5 : center + 5],
            rtol=1e-2,
        )

    def test_laplacian_linearity(self):
        """Test linearity of Laplacian."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Two test functions
        theta1 = jnp.sin(grid.x)
        theta2 = jnp.cos(2 * grid.y)
        a, b = 2.0, -3.0

        # Laplacian of linear combination
        theta_combined = a * theta1 + b * theta2
        theta_combined_hat = fft2(theta_combined)
        lap_combined = ifft2(laplacian(theta_combined_hat, grid))

        # Linear combination of Laplacians
        lap1 = ifft2(laplacian(fft2(theta1), grid))
        lap2 = ifft2(laplacian(fft2(theta2), grid))
        expected = a * lap1 + b * lap2

        np.testing.assert_allclose(lap_combined, expected, rtol=1e-8, atol=1e-12)


class TestFractionalLaplacian:
    """Tests for fractional Laplacian operator."""

    def test_fractional_laplacian_alpha2(self):
        """Test that fractional Laplacian with α=2 matches regular Laplacian."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        theta = jnp.sin(3 * grid.x) * jnp.cos(2 * grid.y)
        theta_hat = fft2(theta)

        # Fractional with α=2
        frac_lap = fractional_laplacian(theta_hat, grid, alpha=2.0)

        # Regular Laplacian (note sign difference)
        reg_lap = -laplacian(theta_hat, grid)

        np.testing.assert_allclose(frac_lap, reg_lap, rtol=1e-10)

    def test_fractional_laplacian_eigenfunction(self):
        """Test fractional Laplacian on sine/cosine eigenfunctions."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Use sine/cosine which are eigenfunctions
        kx_mode, ky_mode = 3, 2
        theta = jnp.sin(kx_mode * grid.x) * jnp.cos(ky_mode * grid.y)
        theta_hat = fft2(theta)

        # Test different α values
        for alpha in [0.5, 1.0, 1.5]:
            frac_lap_hat = fractional_laplacian(theta_hat, grid, alpha=alpha)
            frac_lap = ifft2(frac_lap_hat)

            # Expected eigenvalue: |k|^α
            k_mag = np.sqrt(kx_mode**2 + ky_mode**2)
            expected_eigenvalue = k_mag**alpha
            expected = expected_eigenvalue * theta

            # Check the result
            np.testing.assert_allclose(frac_lap, expected, rtol=1e-8, atol=1e-10)

    def test_fractional_laplacian_zero_mode(self):
        """Test that zero mode is handled correctly."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Constant field (only zero mode)
        theta = jnp.ones((N, N))
        theta_hat = fft2(theta)

        for alpha in [0.5, 1.0, 2.0]:
            frac_lap_hat = fractional_laplacian(theta_hat, grid, alpha=alpha)
            frac_lap = ifft2(frac_lap_hat)

            # Should be zero for any α
            np.testing.assert_allclose(frac_lap, 0, atol=1e-10)


class TestPerpendicularGradient:
    """Tests for perpendicular gradient operator."""

    def test_perpendicular_gradient_streamfunction(self):
        """Test velocity from streamfunction."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Streamfunction: sin(x) * sin(y)
        psi = jnp.sin(grid.x) * jnp.sin(grid.y)
        psi_hat = fft2(psi)

        u, v = perpendicular_gradient(psi_hat, grid)

        # Expected: u = -∂y ψ, v = ∂x ψ
        expected_u = -jnp.sin(grid.x) * jnp.cos(grid.y)
        expected_v = jnp.cos(grid.x) * jnp.sin(grid.y)

        np.testing.assert_allclose(u, expected_u, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(v, expected_v, rtol=1e-8, atol=1e-10)

    def test_perpendicular_gradient_divergence_free(self):
        """Test that velocity field is divergence-free."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Random streamfunction
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, shape=(N, N))
        psi_hat = fft2(psi)

        u, v = perpendicular_gradient(psi_hat, grid)

        # Compute divergence: ∂u/∂x + ∂v/∂y
        u_hat = fft2(u)
        v_hat = fft2(v)
        div_hat = 1j * grid.kx * u_hat + 1j * grid.ky * v_hat
        div = ifft2(div_hat)

        # Should be zero (up to numerical precision)
        np.testing.assert_allclose(div, 0, atol=1e-10)


class TestJacobian:
    """Tests for Jacobian (advection) operator."""

    def test_jacobian_linear_flow(self):
        """Test Jacobian with linear velocity field."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Constant velocity field
        u = jnp.ones((N, N))
        v = jnp.zeros((N, N))

        # Test function with known derivative: sin(x)
        theta = jnp.sin(grid.x)

        jac = jacobian(theta, u, v, grid)

        # Expected: u * ∂θ/∂x = 1 * cos(x)
        expected = jnp.cos(grid.x)

        np.testing.assert_allclose(jac, expected, rtol=1e-8, atol=1e-10)

    def test_jacobian_conservation(self):
        """Test that Jacobian conserves integral of θ."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Random fields
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        theta = jax.random.normal(key1, shape=(N, N))
        u = jax.random.normal(key2, shape=(N, N))
        v = jax.random.normal(key3, shape=(N, N))

        # Make velocity divergence-free
        psi_hat = fft2(jax.random.normal(key2, shape=(N, N)))
        u, v = perpendicular_gradient(psi_hat, grid)

        jac = jacobian(theta, u, v, grid)

        # Integral of Jacobian should be zero
        integral = jnp.mean(jac)
        np.testing.assert_allclose(integral, 0, atol=1e-10)

    def test_jacobian_dealiasing(self):
        """Test that Jacobian applies dealiasing."""
        N = 32  # Small grid to test dealiasing
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # High wavenumber fields that should be dealiased
        k_high = N // 2 - 1
        theta = jnp.sin(k_high * grid.x)
        u = jnp.cos(k_high * grid.x)
        v = jnp.zeros((N, N))

        jac = jacobian(theta, u, v, grid)
        jac_hat = fft2(jac)

        # Check that high wavenumbers are suppressed
        # The product creates modes at 2*k_high which should be dealiased
        k_product = 2 * k_high
        if k_product >= N // 2:
            # These modes should be significantly suppressed due to dealiasing
            # Check that the dealiased region has much lower power
            dealiased_power = jnp.abs(jac_hat * (1 - grid.dealias_mask)).max()
            aliased_power = jnp.abs(jac_hat * grid.dealias_mask).max()
            assert dealiased_power < 0.01 * aliased_power


class TestVelocityComputation:
    """Tests for velocity field computation from theta."""

    def test_streamfunction_inversion(self):
        """Test streamfunction computation."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Test with a simple vorticity field
        theta = jnp.sin(3 * grid.x) * jnp.sin(2 * grid.y)
        theta_hat = fft2(theta)

        # For α=1, u = ∇^⊥(-Δ)^(-1/2)θ
        # First get velocity from our function
        u, v = compute_velocity_from_theta(theta_hat, grid, alpha=1.0)

        # Check that velocity is divergence-free
        u_hat = fft2(u)
        v_hat = fft2(v)
        div_hat = 1j * grid.kx * u_hat + 1j * grid.ky * v_hat
        div = ifft2(div_hat)

        np.testing.assert_allclose(div, 0, atol=1e-10)

    def test_velocity_sqg_case(self):
        """Test velocity computation for SQG (α=1)."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Simple theta field
        theta = jnp.sin(2 * grid.x) * jnp.sin(2 * grid.y)
        theta_hat = fft2(theta)

        u, v = compute_velocity_from_theta(theta_hat, grid, alpha=1.0)

        # For SQG, u = ∇^⊥(-Δ)^(-1/2)θ
        # For our test function, we can compute this analytically
        k_mag = 2 * np.sqrt(2)  # |k| for mode (2,2)
        psi = theta / k_mag  # (-Δ)^(-1/2)θ

        # Velocity from perpendicular gradient
        expected_u = -2 * jnp.sin(2 * grid.x) * jnp.cos(2 * grid.y) / k_mag
        expected_v = 2 * jnp.cos(2 * grid.x) * jnp.sin(2 * grid.y) / k_mag

        np.testing.assert_allclose(u, expected_u, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(v, expected_v, rtol=1e-6, atol=1e-10)

    def test_velocity_divergence_free(self):
        """Test that computed velocity is divergence-free."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Random theta field
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, shape=(N, N))
        theta_hat = fft2(theta)

        # Test for different α values
        for alpha in [0.5, 1.0, 1.5]:
            u, v = compute_velocity_from_theta(theta_hat, grid, alpha=alpha)

            # Compute divergence
            u_hat = fft2(u)
            v_hat = fft2(v)
            div_hat = 1j * grid.kx * u_hat + 1j * grid.ky * v_hat
            div = ifft2(div_hat)

            # Should be zero
            np.testing.assert_allclose(div, 0, atol=1e-10)


class TestOperatorAccuracy:
    """High-order accuracy tests."""

    def test_derivative_convergence(self):
        """Test spectral convergence of derivatives."""
        L = 2 * np.pi
        errors = []
        Ns = [16, 32, 64, 128]

        for N in Ns:
            grid = make_grid(N=N, L=L)

            # Smooth test function
            theta = jnp.exp(jnp.cos(grid.x) + jnp.sin(grid.y))
            theta_hat = fft2(theta)

            # Compute derivative
            dtheta_dx, _ = gradient(theta_hat, grid)

            # Analytical derivative
            dtheta_dx_exact = -theta * jnp.sin(grid.x)

            # Error
            error = jnp.sqrt(jnp.mean((dtheta_dx - dtheta_dx_exact) ** 2))
            errors.append(float(error))

        # For spectral methods, errors should decrease rapidly until machine precision
        # Check that error decreases significantly for first refinement
        assert errors[1] < errors[0] / 100  # 16->32
        # After that we hit machine precision
        assert all(e < 1e-10 for e in errors[1:])  # Should be at machine precision


if __name__ == "__main__":
    pytest.main([__file__])

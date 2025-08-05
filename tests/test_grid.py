"""
Tests for grid module.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pygsquig.core.grid import Grid, make_grid, fft2, ifft2


class TestMakeGrid:
    """Tests for make_grid function."""

    def test_basic_properties(self):
        """Test basic grid properties."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        assert grid.N == N
        assert grid.L == L
        assert grid.x.shape == (N, N)
        assert grid.y.shape == (N, N)
        assert grid.kx.shape == (N, N)
        assert grid.ky.shape == (N, N)
        assert grid.k2.shape == (N, N)
        assert grid.dealias_mask.shape == (N, N)

    def test_odd_N_raises(self):
        """Test that odd N raises ValueError."""
        with pytest.raises(ValueError, match="N must be even"):
            make_grid(N=63, L=2 * np.pi)

    def test_physical_coordinates(self):
        """Test physical space coordinates."""
        N = 4
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Check x coordinates along first row
        expected_x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        np.testing.assert_allclose(grid.x[:, 0], expected_x, rtol=1e-10)

        # Check y coordinates along first column
        np.testing.assert_allclose(grid.y[0, :], expected_x, rtol=1e-10)

    def test_wavenumber_ordering(self):
        """Test wavenumber array ordering for FFT compatibility."""
        N = 4
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Expected wavenumbers for N=4: [0, 1, -2, -1]
        expected_k = np.array([0, 1, -2, -1])
        np.testing.assert_array_equal(grid.kx[:, 0], expected_k)
        np.testing.assert_array_equal(grid.ky[0, :], expected_k)

    def test_nyquist_frequency(self):
        """Test Nyquist frequency handling."""
        N = 8
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Nyquist frequency should be at index N//2
        nyquist_idx = N // 2
        assert grid.kx[nyquist_idx, 0] == -N // 2
        assert grid.ky[0, nyquist_idx] == -N // 2

    def test_k2_calculation(self):
        """Test k² calculation."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Verify k2 = kx² + ky²
        expected_k2 = grid.kx**2 + grid.ky**2
        np.testing.assert_allclose(grid.k2, expected_k2, rtol=1e-10)

        # Check that k2[0, 0] = 0 (zero mode)
        assert grid.k2[0, 0] == 0

    def test_dealias_mask(self):
        """Test 2/3 dealiasing mask."""
        N = 12
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Maximum wavenumber is π * N / L = 6
        # Dealias cutoff should be (2/3) * k_max = 4
        # So modes with |k| >= 4 should be masked out

        # Check zero mode is not masked
        assert grid.dealias_mask[0, 0] == True

        # Check that high wavenumbers are masked
        # k[N//2, 0] corresponds to kx = -N/2 = -6 (in units of 2π/L)
        assert grid.dealias_mask[N // 2, 0] == False

        # Count kept modes - should be less than (2/3)² due to discrete grid
        kept_fraction = grid.dealias_mask.sum() / N**2
        assert 0.25 < kept_fraction < 0.45  # Roughly (2/3)² but accounting for discrete effects


class TestFFTFunctions:
    """Tests for FFT wrapper functions."""

    def test_fft_ifft_roundtrip(self):
        """Test that ifft2(fft2(field)) = field."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Test with a simple field
        field = jnp.sin(grid.x) * jnp.cos(2 * grid.y)
        field_hat = fft2(field)
        field_back = ifft2(field_hat)

        np.testing.assert_allclose(field_back, field, rtol=1e-14, atol=1e-14)

    def test_parseval_theorem(self):
        """Test Parseval's theorem: ⟨|f|²⟩ = (1/N²) ⟨|f̂|²⟩."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Random field
        key = jax.random.PRNGKey(42)
        field = jax.random.normal(key, shape=(N, N))

        # Compute energy in physical and spectral space
        energy_phys = jnp.mean(field**2)
        field_hat = fft2(field)
        # JAX FFT convention: need 1/N² normalization
        energy_spec = jnp.mean(jnp.abs(field_hat) ** 2) / N**2

        np.testing.assert_allclose(energy_phys, energy_spec, rtol=1e-10)

    def test_derivative_via_fft(self):
        """Test spectral derivatives against analytical results."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Test function: sin(x)
        field = jnp.sin(grid.x)
        field_hat = fft2(field)

        # Spectral derivative: ∂x → ikx
        dfdx_hat = 1j * grid.kx * field_hat
        dfdx = ifft2(dfdx_hat)

        # Analytical derivative: cos(x)
        dfdx_exact = jnp.cos(grid.x)

        np.testing.assert_allclose(dfdx, dfdx_exact, rtol=1e-8, atol=1e-10)

    def test_laplacian_eigenfunction(self):
        """Test Laplacian on eigenfunctions."""
        N = 128
        L = 2 * np.pi
        grid = make_grid(N=N, L=L)

        # Eigenfunction of Laplacian: sin(kx*x) * sin(ky*y)
        kx_mode = 3
        ky_mode = 2
        field = jnp.sin(kx_mode * grid.x) * jnp.sin(ky_mode * grid.y)
        field_hat = fft2(field)

        # Apply Laplacian: -k²
        lap_field_hat = -grid.k2 * field_hat
        lap_field = ifft2(lap_field_hat)

        # Expected: -(kx² + ky²) * field
        eigenvalue = -(kx_mode**2 + ky_mode**2)
        expected = eigenvalue * field

        np.testing.assert_allclose(lap_field, expected, rtol=1e-8, atol=1e-10)


class TestGridSmallSizes:
    """Edge case tests with very small grids."""

    def test_N2_grid(self):
        """Test minimal grid with N=2."""
        grid = make_grid(N=2, L=2 * np.pi)

        # Check basic properties
        assert grid.N == 2
        assert grid.x.shape == (2, 2)

        # Wavenumbers should be [0, -1]
        expected_k = np.array([0, -1])
        np.testing.assert_array_equal(grid.kx[:, 0], expected_k)

    def test_N4_grid(self):
        """Test small grid with N=4."""
        grid = make_grid(N=4, L=1.0)

        # Check grid spacing
        dx = 1.0 / 4
        assert np.allclose(grid.x[1, 0] - grid.x[0, 0], dx)

        # Check dealiasing mask keeps only DC and first mode
        # For N=4, dealias cutoff = (2/3)*2 ≈ 1.33
        assert grid.dealias_mask[0, 0] == True  # k=0
        assert grid.dealias_mask[0, 1] == True  # k=1
        assert grid.dealias_mask[0, 2] == False  # k=-2


class TestGridConvergence:
    """Convergence tests for spectral accuracy."""

    def test_spectral_convergence(self):
        """Test spectral convergence for smooth functions."""
        L = 2 * np.pi
        errors = []
        Ns = [16, 32, 64, 128]

        for N in Ns:
            grid = make_grid(N=N, L=L)

            # Simpler smooth test function to avoid aliasing
            # Use low wavenumber modes only
            k_test = 3
            field = jnp.sin(k_test * grid.x) * jnp.cos(k_test * grid.y)

            # Compute spectral derivative
            field_hat = fft2(field)
            dfdx_hat = 1j * grid.kx * field_hat
            dfdx_spectral = ifft2(dfdx_hat)

            # Analytical derivative
            dfdx_exact = k_test * jnp.cos(k_test * grid.x) * jnp.cos(k_test * grid.y)

            # Compute error
            error = jnp.sqrt(jnp.mean((dfdx_spectral - dfdx_exact) ** 2))
            errors.append(float(error))

        # For a function with only low wavenumbers, spectral method should be exact
        # (up to machine precision) once N is large enough
        # Errors should be at machine precision level
        assert all(error < 1e-12 for error in errors[-2:])  # Last two should be very small


if __name__ == "__main__":
    pytest.main([__file__])

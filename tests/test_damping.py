"""
Tests for damping module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import fft2, ifft2, make_grid
from pygsquig.forcing.damping import CombinedDamping, hyperviscosity, large_scale_damping


class TestLargeScaleDamping:
    """Tests for large-scale damping."""

    def test_basic_damping(self):
        """Test basic large-scale damping functionality."""
        grid = make_grid(N=64, L=2 * np.pi)
        mu = 0.1
        kf = 20.0

        # Create a field with low wavenumber
        theta = jnp.sin(2 * grid.x)  # k=2, should be damped
        theta_hat = fft2(theta)

        damping = large_scale_damping(theta_hat, grid, mu, kf)

        # Check that damping is applied in spectral space
        # For modes with k < kf/2, damping should be -mu * theta_hat
        k_mag = jnp.sqrt(grid.k2)
        mask = k_mag < kf / 2

        # Check damped modes
        damped_modes = damping[mask]
        expected_modes = -mu * theta_hat[mask]
        np.testing.assert_allclose(damped_modes, expected_modes, rtol=1e-10)

        # Check undamped modes
        undamped_modes = damping[~mask]
        np.testing.assert_allclose(undamped_modes, 0, atol=1e-10)

    def test_zero_mu(self):
        """Test that mu=0 gives no damping."""
        grid = make_grid(N=64, L=2 * np.pi)

        theta_hat = fft2(jnp.ones((64, 64)))
        damping = large_scale_damping(theta_hat, grid, mu=0.0, kf=20.0)

        assert jnp.allclose(damping, 0)

    def test_mode_selection(self):
        """Test that only large-scale modes are damped."""
        grid = make_grid(N=64, L=2 * np.pi)
        mu = 0.5
        kf = 10.0

        # Low k mode (should be damped)
        theta_low = jnp.sin(2 * grid.x)
        theta_low_hat = fft2(theta_low)
        damping_low = large_scale_damping(theta_low_hat, grid, mu, kf)

        # High k mode (should not be damped)
        theta_high = jnp.sin(20 * grid.x)
        theta_high_hat = fft2(theta_high)
        damping_high = large_scale_damping(theta_high_hat, grid, mu, kf)

        # Low k should have damping
        assert jnp.abs(damping_low).max() > 0

        # High k should have no damping
        assert jnp.allclose(damping_high, 0, atol=1e-10)

    def test_cutoff_scale(self):
        """Test damping cutoff at kf/2."""
        grid = make_grid(N=128, L=2 * np.pi)
        mu = 0.2
        kf = 20.0

        # Test modes around kf/2 = 10
        k_test_values = [5, 8, 10, 12, 15]

        for k in k_test_values:
            theta = jnp.sin(k * grid.x)
            theta_hat = fft2(theta)
            damping = large_scale_damping(theta_hat, grid, mu, kf)

            if k < kf / 2:
                # Should be damped
                assert jnp.abs(damping).max() > 0
            else:
                # Should not be damped
                assert jnp.allclose(damping, 0, atol=1e-10)


class TestHyperviscosity:
    """Tests for hyperviscosity."""

    def test_basic_hyperviscosity(self):
        """Test basic hyperviscosity functionality."""
        grid = make_grid(N=64, L=2 * np.pi)
        nu_p = 0.01
        p = 2

        # Single mode
        k_mode = 5
        theta = jnp.sin(k_mode * grid.x)
        theta_hat = fft2(theta)

        dissipation = hyperviscosity(theta_hat, grid, nu_p, p)

        # Check specific mode
        # Find where the mode is
        idx = jnp.where(jnp.abs(theta_hat) > 1e-10)

        for i, j in zip(idx[0], idx[1]):
            k2 = grid.k2[i, j]
            expected = -nu_p * k2 * theta_hat[i, j]
            np.testing.assert_allclose(dissipation[i, j], expected, rtol=1e-10)

    def test_different_orders(self):
        """Test hyperviscosity with different orders."""
        grid = make_grid(N=64, L=2 * np.pi)
        nu_p = 1e-8

        theta = jnp.sin(10 * grid.x)
        theta_hat = fft2(theta)

        # Test p=2, 4, 8
        dissipations = []
        for p in [2, 4, 8]:
            diss = hyperviscosity(theta_hat, grid, nu_p, p)
            dissipations.append(jnp.abs(diss).max())

        # Higher order should give stronger dissipation for same mode
        assert dissipations[2] > dissipations[1] > dissipations[0]

    def test_zero_nu(self):
        """Test that nu_p=0 gives no dissipation."""
        grid = make_grid(N=64, L=2 * np.pi)

        theta_hat = fft2(jnp.ones((64, 64)))
        dissipation = hyperviscosity(theta_hat, grid, nu_p=0.0, p=8)

        assert jnp.allclose(dissipation, 0)


class TestCombinedDamping:
    """Tests for combined damping class."""

    def test_initialization(self):
        """Test CombinedDamping initialization."""
        damping = CombinedDamping(mu=0.1, kf=20.0, nu_p=1e-8, p=8)

        assert damping.mu == 0.1
        assert damping.kf == 20.0
        assert damping.nu_p == 1e-8
        assert damping.p == 8

    def test_invalid_p(self):
        """Test that invalid p raises error."""
        with pytest.raises(ValueError, match="p must be 2, 4, or 8"):
            CombinedDamping(p=3)

    def test_combined_application(self):
        """Test combined damping application."""
        grid = make_grid(N=64, L=2 * np.pi)
        damping = CombinedDamping(mu=0.1, kf=20.0, nu_p=0.01, p=2)

        # Field with both low and high k modes
        theta = jnp.sin(2 * grid.x) + jnp.sin(25 * grid.x)
        theta_hat = fft2(theta)

        total_damping = damping(theta_hat, grid)

        # Check separately
        large_scale = large_scale_damping(theta_hat, grid, 0.1, 20.0)
        small_scale = hyperviscosity(theta_hat, grid, 0.01, 2)
        expected = large_scale + small_scale

        np.testing.assert_allclose(total_damping, expected, rtol=1e-10)

    def test_only_large_scale(self):
        """Test with only large-scale damping."""
        grid = make_grid(N=64, L=2 * np.pi)
        damping = CombinedDamping(mu=0.5, kf=20.0, nu_p=0.0)

        theta = jnp.sin(3 * grid.x)
        theta_hat = fft2(theta)

        result = damping(theta_hat, grid)
        expected = large_scale_damping(theta_hat, grid, 0.5, 20.0)

        np.testing.assert_allclose(result, expected)

    def test_only_hyperviscosity(self):
        """Test with only hyperviscosity."""
        grid = make_grid(N=64, L=2 * np.pi)
        damping = CombinedDamping(mu=0.0, nu_p=1e-6, p=4)

        theta = jnp.sin(15 * grid.x)
        theta_hat = fft2(theta)

        result = damping(theta_hat, grid)
        expected = hyperviscosity(theta_hat, grid, 1e-6, 4)

        np.testing.assert_allclose(result, expected)


class TestEnergyDissipation:
    """Tests for energy dissipation properties."""

    def test_large_scale_dissipation_positive(self):
        """Test that large-scale damping dissipates energy."""
        grid = make_grid(N=64, L=2 * np.pi)
        mu = 0.1
        kf = 20.0

        # Low-k field
        theta = jnp.sin(3 * grid.x) + jnp.cos(4 * grid.y)
        theta_hat = fft2(theta)

        damping = large_scale_damping(theta_hat, grid, mu, kf)
        damping_phys = ifft2(damping)

        # Energy dissipation rate: -<θ * damping>
        dissipation_rate = -jnp.mean(theta * damping_phys)

        # Should be positive (energy removed)
        assert dissipation_rate > 0

    def test_hyperviscosity_dissipation_positive(self):
        """Test that hyperviscosity dissipates energy."""
        grid = make_grid(N=64, L=2 * np.pi)
        nu_p = 0.001
        p = 2

        # High-k field
        theta = jnp.sin(20 * grid.x) + jnp.cos(15 * grid.y)
        theta_hat = fft2(theta)

        dissipation = hyperviscosity(theta_hat, grid, nu_p, p)
        dissipation_phys = ifft2(dissipation)

        # Energy dissipation rate
        dissipation_rate = -jnp.mean(theta * dissipation_phys)

        # Should be positive
        assert dissipation_rate > 0


class TestDiagnostics:
    """Tests for diagnostic computations."""

    def test_diagnostics_computation(self):
        """Test diagnostic values."""
        grid = make_grid(N=64, L=2 * np.pi)
        damping = CombinedDamping(mu=0.2, kf=20.0, nu_p=1e-6, p=4)

        # Mixed scale field
        theta = jnp.sin(3 * grid.x) + 0.5 * jnp.sin(25 * grid.x)
        theta_hat = fft2(theta)

        diag = damping.get_diagnostics(theta_hat, grid)

        # Check all fields present
        assert "large_scale_dissipation" in diag
        assert "small_scale_dissipation" in diag
        assert "total_dissipation" in diag
        assert "n_large_scale_modes" in diag
        assert "mu" in diag
        assert "nu_p" in diag

        # Check values
        assert diag["mu"] == 0.2
        assert diag["nu_p"] == 1e-6
        assert diag["n_large_scale_modes"] > 0

        # Both dissipations should be positive
        assert diag["large_scale_dissipation"] > 0
        assert diag["small_scale_dissipation"] > 0

        # Total should be sum
        total_expected = diag["large_scale_dissipation"] + diag["small_scale_dissipation"]
        np.testing.assert_allclose(diag["total_dissipation"], total_expected)

    def test_diagnostics_zero_field(self):
        """Test diagnostics with zero field."""
        grid = make_grid(N=64, L=2 * np.pi)
        damping = CombinedDamping(mu=0.1, kf=20.0, nu_p=1e-8, p=8)

        theta_hat = jnp.zeros((64, 64), dtype=jnp.complex128)
        diag = damping.get_diagnostics(theta_hat, grid)

        # All dissipations should be zero
        assert diag["large_scale_dissipation"] == 0.0
        assert diag["small_scale_dissipation"] == 0.0
        assert diag["total_dissipation"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])

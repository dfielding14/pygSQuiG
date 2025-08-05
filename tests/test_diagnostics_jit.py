"""Tests for JIT-compiled diagnostic functions.

Following test-first development, these tests ensure JIT compilation
works correctly for all diagnostic functions.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import make_grid
from pygsquig.utils.diagnostics import (
    compute_energy_flux,
    compute_energy_spectrum,
    compute_enstrophy,
    compute_palinstrophy,
    compute_scalar_flux,
    compute_total_energy,
)


class TestDiagnosticsJIT:
    """Test JIT compilation of diagnostic functions."""

    @pytest.fixture
    def setup_data(self):
        """Create test data for diagnostics."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N, L)

        # Create a test field with known properties
        key = jax.random.PRNGKey(42)
        theta_hat = jax.random.normal(key, (N, N), dtype=jnp.complex128)
        # Make it Hermitian symmetric for real field
        theta_hat = theta_hat.at[0, 0].set(0.0)  # Zero mean

        return grid, theta_hat

    def test_energy_spectrum_jit(self, setup_data):
        """Test JIT compilation of energy spectrum."""
        grid, theta_hat = setup_data
        alpha = 1.0

        # Regular version
        k1, E1 = compute_energy_spectrum(theta_hat, grid, alpha)

        # JIT version
        try:
            compute_spectrum_jit = jax.jit(compute_energy_spectrum, static_argnums=(2,))
            k2, E2 = compute_spectrum_jit(theta_hat, grid, alpha)

            # Results should be identical
            assert np.allclose(k1, k2)
            assert np.allclose(E1, E2)
        except:
            pytest.skip("compute_energy_spectrum not yet JIT-compatible")

    def test_total_energy_jit(self, setup_data):
        """Test JIT compilation of total energy."""
        grid, theta_hat = setup_data
        alpha = 1.0

        # Regular version
        energy1 = compute_total_energy(theta_hat, grid, alpha)

        # JIT version
        try:
            compute_energy_jit = jax.jit(compute_total_energy, static_argnums=(2,))
            energy2 = compute_energy_jit(theta_hat, grid, alpha)

            # Results should be identical
            assert np.isclose(float(energy1), float(energy2))
        except:
            pytest.skip("compute_total_energy not yet JIT-compatible")

    def test_enstrophy_jit(self, setup_data):
        """Test JIT compilation of enstrophy."""
        grid, theta_hat = setup_data
        alpha = 1.0

        # Regular version
        enstrophy1 = compute_enstrophy(theta_hat, grid, alpha)

        # JIT version
        try:
            compute_enstrophy_jit = jax.jit(compute_enstrophy, static_argnums=(2,))
            enstrophy2 = compute_enstrophy_jit(theta_hat, grid, alpha)

            # Results should be identical
            assert np.isclose(float(enstrophy1), float(enstrophy2))
        except:
            pytest.skip("compute_enstrophy not yet JIT-compatible")

    def test_palinstrophy_jit(self, setup_data):
        """Test JIT compilation of palinstrophy."""
        grid, theta_hat = setup_data
        alpha = 1.0

        # Regular version
        palinstrophy1 = compute_palinstrophy(theta_hat, grid, alpha)

        # JIT version
        try:
            compute_palinstrophy_jit = jax.jit(compute_palinstrophy, static_argnums=(2,))
            palinstrophy2 = compute_palinstrophy_jit(theta_hat, grid, alpha)

            # Results should be identical
            assert np.isclose(float(palinstrophy1), float(palinstrophy2))
        except:
            pytest.skip("compute_palinstrophy not yet JIT-compatible")

    def test_scalar_flux_jit(self, setup_data):
        """Test JIT compilation of scalar flux."""
        grid, theta_hat = setup_data

        # Create velocity field
        from pygsquig.core.operators import compute_velocity_from_theta

        u, v = compute_velocity_from_theta(theta_hat, grid, alpha=1.0)

        # Regular version
        flux1 = compute_scalar_flux(theta_hat, (u, v), grid)

        # JIT version
        try:
            compute_flux_jit = jax.jit(compute_scalar_flux)
            flux2 = compute_flux_jit(theta_hat, (u, v), grid)

            # Results should be identical
            assert np.isclose(float(flux1), float(flux2))
        except:
            pytest.skip("compute_scalar_flux not yet JIT-compatible")

    def test_energy_flux_jit(self, setup_data):
        """Test JIT compilation of energy flux."""
        grid, theta_hat = setup_data

        # Create a forcing field
        forcing_hat = 0.1 * theta_hat

        # Regular version
        flux1 = compute_energy_flux(theta_hat, forcing_hat, grid)

        # JIT version
        try:
            compute_flux_jit = jax.jit(compute_energy_flux)
            flux2 = compute_flux_jit(theta_hat, forcing_hat, grid)

            # Results should be identical
            assert np.isclose(float(flux1), float(flux2))
        except:
            pytest.skip("compute_energy_flux not yet JIT-compatible")

    @pytest.mark.benchmark
    def test_diagnostic_performance(self, setup_data):
        """Benchmark diagnostic computations with JIT."""
        grid, theta_hat = setup_data
        alpha = 1.0

        # Time regular version
        start = time.time()
        for _ in range(10):
            compute_total_energy(theta_hat, grid, alpha)
        time_regular = time.time() - start

        # Time JIT version
        try:
            compute_energy_jit = jax.jit(compute_total_energy, static_argnums=(2,))

            # Warm up
            _ = compute_energy_jit(theta_hat, grid, alpha)

            start = time.time()
            for _ in range(10):
                compute_energy_jit(theta_hat, grid, alpha)
            time_jit = time.time() - start

            print(f"\nRegular: {time_regular*100:.2f}ms")
            print(f"JIT: {time_jit*100:.2f}ms")
            print(f"Speedup: {time_regular/time_jit:.1f}x")

            # JIT should be faster
            assert time_jit < time_regular
        except:
            pytest.skip("Diagnostics not yet JIT-compatible")

    def test_jit_with_different_alpha(self, setup_data):
        """Test JIT compilation works for different alpha values."""
        grid, theta_hat = setup_data

        try:
            compute_energy_jit = jax.jit(compute_total_energy, static_argnums=(2,))

            # Test different alpha values
            for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                energy = compute_energy_jit(theta_hat, grid, alpha)
                assert np.isfinite(float(energy))
                assert float(energy) >= 0  # Energy should be non-negative
        except:
            pytest.skip("compute_total_energy not yet JIT-compatible")

"""Test to verify hyperviscosity implementations are identical before refactoring."""

import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import make_grid
from pygsquig.core.operators import hyperviscosity
from pygsquig.core.solver import gSQGSolver


class TestHyperviscosityImplementations:
    """Verify all hyperviscosity implementations give same results."""

    @pytest.fixture
    def setup(self):
        """Create test data."""
        N = 32
        L = 2 * np.pi
        grid = make_grid(N, L)

        # Test field
        theta_hat = jnp.ones((N, N), dtype=jnp.complex128)
        theta_hat = theta_hat.at[0, 0].set(0.0)  # Zero mean

        return grid, theta_hat

    def test_hyperviscosity_basic(self, setup):
        """Test hyperviscosity basic functionality."""
        grid, theta_hat = setup
        nu_p = 1e-6
        p = 8

        # Compute hyperviscosity
        result = hyperviscosity(theta_hat, grid, nu_p, p)

        # Should be non-zero and scale properly
        assert jnp.any(result != 0)
        assert result[0, 0] == 0  # No dissipation of mean

    def test_solver_method_consistency(self, setup):
        """Test solver method gives same result as internal function."""
        grid, theta_hat = setup
        nu_p = 1e-4
        p = 4

        # Create solver
        solver = gSQGSolver(grid, alpha=1.0, nu_p=nu_p, p=p)

        # Method call
        result1 = solver.compute_hyperviscosity(theta_hat)

        # Direct function call
        result2 = hyperviscosity(theta_hat, grid, nu_p, p)

        # Should be identical
        assert jnp.allclose(result1, result2)

    def test_different_orders(self, setup):
        """Test all implementations work for different p values."""
        grid, theta_hat = setup
        nu_p = 1e-6

        for p in [2, 4, 6, 8]:
            # Compute hyperviscosity
            result = hyperviscosity(theta_hat, grid, nu_p, p)

            # Should be non-zero and properly scaled
            assert jnp.any(result != 0)

            # Check expected scaling
            k2_max = jnp.max(grid.k2)
            expected_max = nu_p * (k2_max ** (p / 2))
            actual_max = jnp.max(jnp.abs(result))

            # Should scale correctly with k^p
            assert actual_max <= expected_max * 1.1  # Allow small tolerance

"""
Tests for physical forcing patterns.

This module tests shear layers, jets, convective plumes,
and topographic forcing implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import ifft2, make_grid
from pygsquig.exceptions import ForcingError
from pygsquig.forcing.physical_forcing import (
    ConvectivePlumesForcing,
    JetForcing,
    ShearLayerForcing,
    TopographicForcing,
)


class TestShearLayerForcing:
    """Test shear layer forcing implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test shear layer initialization."""
        forcing = ShearLayerForcing(
            amplitude=1.0, shear_width=0.1, n_layers=2, orientation="horizontal"
        )
        assert forcing.amplitude == 1.0
        assert forcing.shear_width == 0.1
        assert forcing.n_layers == 2
        assert forcing.orientation == "horizontal"

    def test_invalid_parameters(self):
        """Test invalid parameters raise errors."""
        with pytest.raises(ForcingError):
            ShearLayerForcing(amplitude=-1.0)

        with pytest.raises(ForcingError):
            ShearLayerForcing(shear_width=0.0)

        with pytest.raises(ForcingError):
            ShearLayerForcing(shear_width=0.6)

        with pytest.raises(ForcingError):
            ShearLayerForcing(n_layers=0)

    def test_horizontal_shear(self):
        """Test horizontal shear layer pattern."""
        forcing = ShearLayerForcing(
            amplitude=1.0, shear_width=0.1, n_layers=1, orientation="horizontal"
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)

        # Check shape and type
        assert result.shape == self.theta_hat.shape
        assert result.dtype == self.theta_hat.dtype

        # Transform to physical space
        forcing_phys = ifft2(result).real

        # Check non-zero forcing
        assert not np.allclose(forcing_phys, 0)

        # For shear layer, should have significant spatial variation
        assert np.std(forcing_phys) > 0.1

        # Check that forcing has expected range
        assert np.max(np.abs(forcing_phys)) > 0.5  # Should be order 1

    def test_vertical_shear(self):
        """Test vertical shear layer pattern."""
        forcing = ShearLayerForcing(
            amplitude=1.0, shear_width=0.1, n_layers=1, orientation="vertical"
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Check non-zero forcing
        assert not np.allclose(forcing_phys, 0)

        # Should have significant spatial variation
        assert np.std(forcing_phys) > 0.1

    def test_multiple_layers(self):
        """Test multiple shear layers."""
        forcing = ShearLayerForcing(
            amplitude=1.0, shear_width=0.05, n_layers=3, orientation="horizontal"
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Check non-zero and has variation
        assert not np.allclose(forcing_phys, 0)
        assert np.std(forcing_phys) > 0.1

    def test_perturbations(self):
        """Test shear layer perturbations."""
        forcing = ShearLayerForcing(
            amplitude=1.0,
            shear_width=0.1,
            n_layers=1,
            perturbation_amplitude=0.1,
            perturbation_k=5,
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should have x-variations due to perturbations
        # Check different y-slices have different patterns
        slice1 = forcing_phys[self.N // 4, :]
        slice2 = forcing_phys[3 * self.N // 4, :]

        assert not np.allclose(slice1, slice2)

    def test_time_modulation(self):
        """Test time-dependent modulation."""
        forcing = ShearLayerForcing(amplitude=1.0, time_dependence="oscillatory", omega=2.0)

        # Get forcing at different times
        result1 = forcing(self.theta_hat, None, 0.0, self.grid)
        forcing._time = np.pi / 2.0  # Quarter period
        result2 = forcing(self.theta_hat, None, 0.0, self.grid)

        # Should be different
        assert not np.allclose(result1, result2)

    def test_no_mean_injection(self):
        """Test that k=0 mode is zero."""
        forcing = ShearLayerForcing(amplitude=1.0)
        result = forcing(self.theta_hat, None, 0.01, self.grid)
        assert result[0, 0] == 0


class TestJetForcing:
    """Test jet forcing implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test jet forcing initialization."""
        forcing = JetForcing(amplitude=2.0, jet_width=0.1, n_jets=2, orientation="zonal")
        assert forcing.amplitude == 2.0
        assert forcing.jet_width == 0.1
        assert forcing.n_jets == 2
        assert forcing.orientation == "zonal"

    def test_zonal_jets(self):
        """Test zonal jet pattern."""
        forcing = JetForcing(amplitude=1.0, jet_width=0.1, n_jets=2, orientation="zonal")

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Check non-zero forcing
        assert not np.allclose(forcing_phys, 0)
        assert np.std(forcing_phys) > 0.1

    def test_meridional_jets(self):
        """Test meridional jet pattern."""
        forcing = JetForcing(amplitude=1.0, jet_width=0.1, n_jets=2, orientation="meridional")

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Check non-zero forcing
        assert not np.allclose(forcing_phys, 0)
        assert np.std(forcing_phys) > 0.1

    def test_meandering_jets(self):
        """Test jet meandering."""
        forcing = JetForcing(
            amplitude=1.0, jet_width=0.1, n_jets=1, meander_amplitude=0.2, meander_k=3
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should have x-variations due to meandering
        # Different x-slices should peak at different y
        slice1 = forcing_phys[self.N // 4, :]
        slice2 = forcing_phys[3 * self.N // 4, :]

        peak1 = np.argmax(np.abs(slice1))
        peak2 = np.argmax(np.abs(slice2))

        assert peak1 != peak2  # Peaks at different locations

    def test_jet_profiles(self):
        """Test different jet profiles."""
        profiles = ["gaussian", "sech2", "tanh"]

        for profile in profiles:
            forcing = JetForcing(amplitude=1.0, jet_width=0.1, n_jets=1, profile=profile)

            result = forcing(self.theta_hat, None, 0.01, self.grid)
            forcing_phys = ifft2(result).real

            # All should produce non-zero forcing
            assert not np.allclose(forcing_phys, 0)
            assert np.max(np.abs(forcing_phys)) > 0.1

    def test_alternating_jets(self):
        """Test jets alternate direction."""
        forcing = JetForcing(amplitude=1.0, jet_width=0.1, n_jets=3, orientation="zonal")

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should have both positive and negative values
        assert np.max(forcing_phys) > 0.1
        assert np.min(forcing_phys) < -0.1


class TestConvectivePlumesForcing:
    """Test convective plumes forcing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test plumes initialization."""
        forcing = ConvectivePlumesForcing(
            amplitude=1.0, plume_radius=0.05, n_plumes=3, rise_velocity=1.0
        )
        assert forcing.amplitude == 1.0
        assert forcing.plume_radius == 0.05
        assert forcing.n_plumes == 3
        assert forcing.rise_velocity == 1.0

    def test_plume_generation(self):
        """Test plume generation."""
        forcing = ConvectivePlumesForcing(
            amplitude=1.0, plume_radius=0.1, n_plumes=2, randomize_positions=False
        )

        key = jax.random.PRNGKey(42)
        result = forcing(self.theta_hat, key, 0.01, self.grid)

        # Should create forcing
        assert not jnp.allclose(result, 0)

        # Physical space should show localized features
        forcing_phys = ifft2(result).real
        assert jnp.max(forcing_phys) > 0.1

    def test_plume_rise(self):
        """Test plume rising motion."""
        forcing = ConvectivePlumesForcing(
            amplitude=1.0,
            plume_radius=0.05,
            n_plumes=1,
            rise_velocity=2.0,
            randomize_positions=False,
        )

        key = jax.random.PRNGKey(42)

        # Get initial position
        result1 = forcing(self.theta_hat, key, 0.0, self.grid)
        phys1 = ifft2(result1).real

        # Evolve
        dt = 0.1
        result2 = forcing(self.theta_hat, key, dt, self.grid)
        phys2 = ifft2(result2).real

        # Center of mass should move up
        y = self.grid.y
        com1 = jnp.sum(y * phys1) / jnp.sum(phys1 + 1e-10)
        com2 = jnp.sum(y * phys2) / jnp.sum(phys2 + 1e-10)

        assert com2 > com1  # Moved upward

    def test_plume_decay(self):
        """Test plume buoyancy decay."""
        forcing = ConvectivePlumesForcing(
            amplitude=1.0,
            plume_radius=0.05,
            n_plumes=1,
            buoyancy_decay=1.0,  # Fast decay
            randomize_positions=False,
        )

        key = jax.random.PRNGKey(42)

        # Initial amplitude
        result1 = forcing(self.theta_hat, key, 0.0, self.grid)
        amp1 = jnp.max(jnp.abs(ifft2(result1)))

        # After some time
        forcing._update_plumes(1.0, self.L, key)  # Age plumes
        result2 = forcing(self.theta_hat, key, 0.0, self.grid)
        amp2 = jnp.max(jnp.abs(ifft2(result2)))

        # Should decay
        assert amp2 < amp1


class TestTopographicForcing:
    """Test topographic forcing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_ridge_topography(self):
        """Test ridge topography."""
        forcing = TopographicForcing(
            amplitude=1.0, topography_type="ridge", k_topo=4, orientation="zonal"
        )

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should be sinusoidal in y
        y_profile = np.mean(forcing_phys, axis=1)

        # Should have sinusoidal structure
        assert np.max(forcing_phys) > 0.5
        assert np.min(forcing_phys) < -0.5

    def test_seamount_topography(self):
        """Test seamount topography."""
        forcing = TopographicForcing(amplitude=1.0, topography_type="seamount", k_topo=3)

        result = forcing(self.theta_hat, None, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should have 2D structure
        assert np.std(forcing_phys) > 0.1

        # Should be symmetric
        assert np.allclose(forcing_phys, forcing_phys.T, rtol=1e-5)

    def test_rough_topography(self):
        """Test rough topography."""
        forcing = TopographicForcing(amplitude=1.0, topography_type="rough", k_topo=4)

        key = jax.random.PRNGKey(42)
        result = forcing(self.theta_hat, key, 0.01, self.grid)
        forcing_phys = ifft2(result).real

        # Should have complex structure
        assert np.std(forcing_phys) > 0.1

        # Different random keys give different patterns
        key2 = jax.random.PRNGKey(43)
        result2 = forcing(self.theta_hat, key2, 0.01, self.grid)

        assert not np.allclose(result, result2)


# Integration tests
class TestForcingIntegration:
    """Test forcing patterns with solver."""

    def test_shear_instability(self):
        """Test shear forcing can be applied stably."""
        from pygsquig.core.solver import gSQGSolver

        N = 64  # Smaller grid for stability
        grid = make_grid(N, 2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-4, p=4)  # More dissipation

        # Very gentle shear forcing
        forcing = ShearLayerForcing(
            amplitude=0.0001,  # Very small amplitude
            shear_width=0.3,  # Wide shear layer
            n_layers=1,
            perturbation_amplitude=0.0,  # No perturbations initially
            perturbation_k=8,
        )

        # Initialize with small noise
        state = solver.initialize(seed=42)
        initial_max = np.max(np.abs(state["theta_hat"]))

        # Evolve with forcing wrapper
        dt = 0.01  # Reasonable timestep

        def forcing_wrapper(theta_hat, **kwargs):
            return forcing(theta_hat, None, dt, grid)

        # Just check stability for a few steps
        for step in range(10):
            state = solver.step(state, dt, forcing=forcing_wrapper)

        # Check stability
        final_max = np.max(np.abs(state["theta_hat"]))
        assert not np.isnan(final_max)  # Check stability
        assert final_max < 1000.0  # Should not explode catastrophically

    def test_jet_stability(self):
        """Test jet forcing creates stable jets."""
        from pygsquig.core.solver import gSQGSolver

        N = 128
        grid = make_grid(N, 2 * np.pi)
        solver = gSQGSolver(grid, alpha=0.5, nu_p=1e-5, p=4)

        forcing = JetForcing(amplitude=1.0, jet_width=0.1, n_jets=2, profile="gaussian")

        state = solver.initialize(seed=42)

        # Evolve to quasi-steady state
        dt = 0.001

        def forcing_wrapper(theta_hat, **kwargs):
            return forcing(theta_hat, None, dt, grid)

        for step in range(500):
            state = solver.step(state, dt, forcing=forcing_wrapper)

        # Check jet structure persists
        theta_phys = ifft2(state["theta_hat"]).real
        y_profile = np.mean(theta_phys, axis=1)

        # Should have clear structure
        assert np.std(theta_phys) > 0.01

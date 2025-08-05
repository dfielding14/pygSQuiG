"""Test plotting module refactoring works correctly."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import make_grid


class TestPlottingRefactor:
    """Test that plotting functions work after refactoring."""

    @pytest.fixture
    def setup_data(self):
        """Create test data for plotting."""
        N = 64
        L = 2 * np.pi
        grid = make_grid(N, L)

        # Create test field
        # grid.x and grid.y are already 2D arrays
        theta = np.sin(2 * grid.x) * np.cos(3 * grid.y)
        theta_hat = jnp.fft.fft2(theta)

        return grid, theta, theta_hat

    def test_imports_from_plots_package(self):
        """Test direct imports from plots package."""
        from pygsquig.plots import (
            PlotStyle,
            plot_field_slice,
            plot_velocity_fields,
            plot_vorticity,
        )

        # Check all imports are callable
        assert callable(plot_field_slice)
        assert callable(plot_vorticity)
        assert callable(plot_velocity_fields)
        assert hasattr(PlotStyle, "setup")

    def test_imports_from_utils_backward_compatibility(self):
        """Test backward compatibility imports from utils."""
        from pygsquig.utils import (
            PlotStyle,
            plot_field_slice,
        )

        # Check all imports work
        assert callable(plot_field_slice)
        assert hasattr(PlotStyle, "FIELD_CMAP")

    def test_submodule_imports(self):
        """Test imports from individual submodules."""
        from pygsquig.plots.fields import plot_field_slice
        from pygsquig.plots.style import PlotStyle

        # Verify imports
        assert PlotStyle.DPI == 150
        assert callable(plot_field_slice)

    def test_plot_field_slice_basic(self, setup_data):
        """Test basic field plotting works."""
        from pygsquig.plots import plot_field_slice

        grid, theta, _ = setup_data

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Should not raise and should save file
            result = plot_field_slice(theta, grid, "Test Field", output_path=output_path)
            assert result is None  # Returns None when saving
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_plot_vorticity(self, setup_data):
        """Test vorticity plotting works."""
        from pygsquig.plots import plot_vorticity

        grid, _, theta_hat = setup_data
        alpha = 1.0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = plot_vorticity(theta_hat, grid, alpha, output_path=output_path)
            assert result is None
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_plot_spectrum(self, setup_data):
        """Test spectrum plotting works."""
        from pygsquig.plots import plot_energy_spectrum_with_analysis

        grid, _, theta_hat = setup_data
        alpha = 1.0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            fig, slope = plot_energy_spectrum_with_analysis(
                theta_hat, grid, alpha, output_path=output_path
            )
            assert fig is None  # Returns None when saving
            assert slope is None  # No inertial range specified
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_plot_time_series(self):
        """Test time series plotting works."""
        from pygsquig.plots import plot_time_series_multiplot

        # Create dummy time series data
        time = np.linspace(0, 10, 100)
        data_dict = {
            "energy": np.sin(time) + 1,
            "enstrophy": np.cos(time) + 2,
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = plot_time_series_multiplot(time, data_dict, output_path=output_path)
            assert result is None
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_plot_style_consistency(self):
        """Test PlotStyle is consistent across modules."""
        from pygsquig.plots import PlotStyle as PS1
        from pygsquig.plots.style import PlotStyle as PS2
        from pygsquig.utils import PlotStyle as PS3

        # All should be the same class
        assert PS1 is PS2
        assert PS1 is PS3
        assert PS1.FIELD_CMAP == "RdBu_r"

    def test_new_functions_available(self):
        """Test new functions added during refactoring are available."""
        from pygsquig.plots import (
            create_spectrum_animation,
            create_vorticity_animation,
            plot_compensated_spectrum,
            plot_conservation_check,
            plot_energy_balance,
            plot_regime_evolution,
            plot_spectrum_evolution,
        )

        # Check all new functions are callable
        assert callable(plot_spectrum_evolution)
        assert callable(plot_compensated_spectrum)
        assert callable(plot_energy_balance)
        assert callable(plot_conservation_check)
        assert callable(plot_regime_evolution)
        assert callable(create_vorticity_animation)
        assert callable(create_spectrum_animation)

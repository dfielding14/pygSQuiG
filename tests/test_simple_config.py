"""Tests for simplified configuration system."""

import tempfile
from pathlib import Path

import pytest

from pygsquig.io.config_adapter import adapt_config
from pygsquig.io.simple_config import Config, load_config


class TestSimpleConfig:
    """Test the simplified Config class."""

    def test_default_config(self):
        """Test creation with default values."""
        config = Config()

        # Check some defaults
        assert config.get("grid.N") == 256
        assert config.get("grid.L") == 2 * 3.141592653589793
        assert config.get("solver.alpha") == 1.0
        assert config.get("forcing.enabled") is False

    def test_custom_config(self):
        """Test creation with custom values."""
        custom = {"grid": {"N": 512, "L": 10.0}, "solver": {"alpha": 0.5}}
        config = Config(custom)

        assert config.get("grid.N") == 512
        assert config.get("grid.L") == 10.0
        assert config.get("solver.alpha") == 0.5
        # Other values should still be defaults
        assert config.get("solver.nu_p") == 1.0e-16

    def test_dot_notation_get_set(self):
        """Test getting and setting with dot notation."""
        config = Config()

        # Test get
        assert config.get("solver.nu_p") == 1.0e-16
        assert config.get("nonexistent.key", "default") == "default"

        # Test set
        config.set("solver.nu_p", 1.0e-12)
        assert config.get("solver.nu_p") == 1.0e-12

        # Test setting new keys
        config.set("custom.nested.value", 42)
        assert config.get("custom.nested.value") == 42

    def test_attribute_access(self):
        """Test attribute-style access to sections."""
        config = Config()

        # Access top-level sections
        assert config.grid["N"] == 256
        assert config.solver["alpha"] == 1.0

        with pytest.raises(AttributeError):
            _ = config.nonexistent_section

    def test_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        Config({"grid": {"N": 128}})

        # Invalid N (not even)
        with pytest.raises(ValueError, match="Invalid grid.N"):
            Config({"grid": {"N": 127}})

        # Invalid alpha (out of range)
        with pytest.raises(ValueError, match="Invalid solver.alpha"):
            Config({"solver": {"alpha": 3.0}})

        # Invalid p value
        with pytest.raises(ValueError, match="Invalid solver.p"):
            Config({"solver": {"p": 6}})

        # Invalid time integration method
        with pytest.raises(ValueError, match="Invalid time_integration.method"):
            Config({"time_integration": {"method": "INVALID"}})

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        custom = {
            "grid": {"N": 512},
            "solver": {"alpha": 0.5, "nu_p": 1e-10},
            "forcing": {"enabled": True, "kf": 30.0},
        }
        config = Config(custom)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            temp_path = Path(f.name)

        try:
            # Load back
            loaded = Config.from_yaml(temp_path)

            assert loaded.get("grid.N") == 512
            assert loaded.get("solver.alpha") == 0.5
            assert loaded.get("solver.nu_p") == 1e-10
            assert loaded.get("forcing.enabled") is True
            assert loaded.get("forcing.kf") == 30.0
        finally:
            temp_path.unlink()

    def test_load_existing_yaml(self):
        """Test loading existing YAML config files."""
        yaml_content = """
grid:
  N: 256
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    nu_p: 1.0e-12
    p: 8

forcing:
  type: ring
  kf: 30.0
  dk: 3.0
  epsilon: 0.5
  tau_f: 0.0

simulation:
  t_end: 50.0
  checkpoint_interval: 5.0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Load using simple config
            config = Config.from_yaml(temp_path)

            # Check values (note: dissipation fields get flattened)
            assert config.get("grid.N") == 256
            assert config.get("solver.alpha") == 1.0
            # The nested dissipation should be handled
            assert config.get("solver.dissipation.nu_p") == 1.0e-12
            assert config.get("solver.dissipation.p") == 8
            assert config.get("forcing.kf") == 30.0
            assert config.get("simulation.t_end") == 50.0
        finally:
            temp_path.unlink()


class TestConfigAdapter:
    """Test the configuration adapter for backward compatibility."""

    def test_basic_adapter(self):
        """Test basic adapter functionality."""
        config = Config(
            {"grid": {"N": 512, "L": 10.0}, "solver": {"alpha": 0.5, "nu_p": 1e-12, "p": 4}}
        )

        adapted = adapt_config(config)

        # Test grid access
        assert adapted.grid.N == 512
        assert adapted.grid.L == 10.0

        # Test solver access
        assert adapted.solver.alpha == 0.5
        assert adapted.solver.dissipation.nu_p == 1e-12
        assert adapted.solver.dissipation.p == 4

    def test_forcing_adapter(self):
        """Test forcing configuration adaptation."""
        # Test with forcing enabled
        config = Config({"forcing": {"type": "ring", "kf": 25.0, "epsilon": 0.3}})

        adapted = adapt_config(config)

        assert adapted.forcing is not None
        assert adapted.forcing.kf == 25.0
        assert adapted.forcing.epsilon == 0.3

        # Test with forcing disabled
        config2 = Config({"forcing": {"enabled": False}})
        adapted2 = adapt_config(config2)
        assert adapted2.forcing is None

    def test_damping_adapter(self):
        """Test damping configuration adaptation."""
        config = Config(
            {
                "damping": {"enabled": True, "mu": 0.5, "k_cutoff_factor": 0.4},
                "forcing": {"kf": 20.0},
            }
        )
        adapted = adapt_config(config)

        assert adapted.solver.damping is not None
        assert adapted.solver.damping.mu == 0.5
        # kf should be forcing.kf * k_cutoff_factor
        assert adapted.solver.damping.kf == 8.0  # 20.0 * 0.4

    def test_time_integration_adapter(self):
        """Test time integration configuration adaptation."""
        config = Config(
            {"time_integration": {"method": "SSP-RK3", "adaptive": False, "dt": 0.002}}
        )
        adapted = adapt_config(config)

        assert adapted.solver.time_integration.method == "SSP-RK3"
        assert adapted.solver.time_integration.adaptive_cfl is False
        assert adapted.solver.time_integration.dt == 0.002

    def test_compatibility_with_run_script(self):
        """Test that adapter works with patterns from run.py."""
        config = Config(
            {
                "grid": {"N": 128},
                "solver": {"alpha": 1.0},
                "forcing": {"type": "ring", "epsilon": 0.1},
                "simulation": {"t_end": 10.0},
            }
        )

        adapted = adapt_config(config)

        # Patterns from run.py
        N = adapted.grid.N
        L = adapted.grid.L
        alpha = adapted.solver.alpha
        nu_p = adapted.solver.dissipation.nu_p
        p = adapted.solver.dissipation.p

        assert N == 128
        assert L > 0
        assert alpha == 1.0
        assert nu_p >= 0
        assert p in [2, 4, 8]

        # Check forcing exists
        if adapted.forcing:
            assert hasattr(adapted.forcing, "epsilon")


class TestBackwardCompatibility:
    """Test that simplified config maintains backward compatibility."""

    def test_load_function(self):
        """Test the load_config compatibility function."""
        yaml_content = """
grid:
  N: 64
solver:
  alpha: 0.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Should work like before
            config = load_config(temp_path)
            assert isinstance(config, Config)
            assert config.get("grid.N") == 64
            assert config.get("solver.alpha") == 0.0
        finally:
            temp_path.unlink()

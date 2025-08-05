"""
Tests for scalar configuration in the configuration system.

This module tests the scalar-specific configuration classes
and their integration with the main configuration system.
"""

import pytest
import yaml
from pathlib import Path
import tempfile

from pygsquig.io.config import (
    ScalarSourceConfig,
    PassiveScalarConfig,
    ScalarsConfig,
    RunConfig,
    GridConfig,
    SolverConfig,
)


class TestScalarSourceConfig:
    """Test the ScalarSourceConfig dataclass."""

    def test_basic_source_config(self):
        """Test basic source configuration."""
        source = ScalarSourceConfig(type="exponential", parameters={"rate": 0.1})
        assert source.type == "exponential"
        assert source.parameters["rate"] == 0.1

    def test_source_types(self):
        """Test all valid source types."""
        valid_types = ["exponential", "localized", "chemical", "periodic", "none"]

        for source_type in valid_types:
            source = ScalarSourceConfig(type=source_type)
            assert source.type == source_type

    def test_invalid_source_type(self):
        """Test that invalid source type raises error."""
        with pytest.raises(ValueError, match="Unknown source type"):
            ScalarSourceConfig(type="invalid_type")

    def test_empty_parameters(self):
        """Test source with empty parameters."""
        source = ScalarSourceConfig(type="exponential")
        assert source.parameters == {}


class TestPassiveScalarConfig:
    """Test the PassiveScalarConfig dataclass."""

    def test_basic_scalar_config(self):
        """Test basic scalar configuration."""
        scalar = PassiveScalarConfig(name="tracer", kappa=0.01)
        assert scalar.name == "tracer"
        assert scalar.kappa == 0.01
        assert scalar.source is None
        assert scalar.initial_condition == "zero"
        assert scalar.initial_params == {}

    def test_scalar_with_source(self):
        """Test scalar with source term."""
        source = ScalarSourceConfig(
            type="localized", parameters={"x0": 3.14, "y0": 3.14, "sigma": 0.5, "amplitude": 1.0}
        )
        scalar = PassiveScalarConfig(
            name="temperature",
            kappa=0.02,
            source=source,
            initial_condition="gaussian",
            initial_params={"center": [3.14, 3.14], "width": 1.0},
        )
        assert scalar.source.type == "localized"
        assert scalar.initial_condition == "gaussian"

    def test_negative_diffusivity(self):
        """Test that negative diffusivity raises error."""
        with pytest.raises(ValueError, match="kappa must be non-negative"):
            PassiveScalarConfig(name="test", kappa=-0.1)

    def test_invalid_initial_condition(self):
        """Test that invalid initial condition raises error."""
        with pytest.raises(ValueError, match="Unknown initial condition"):
            PassiveScalarConfig(name="test", initial_condition="invalid")

    def test_valid_initial_conditions(self):
        """Test all valid initial condition types."""
        valid_conditions = ["zero", "random", "gaussian", "uniform"]

        for condition in valid_conditions:
            scalar = PassiveScalarConfig(name="test", initial_condition=condition)
            assert scalar.initial_condition == condition


class TestScalarsConfig:
    """Test the ScalarsConfig dataclass."""

    def test_disabled_scalars(self):
        """Test disabled scalars configuration."""
        scalars = ScalarsConfig(enabled=False)
        assert not scalars.enabled
        assert scalars.species == []

    def test_single_species(self):
        """Test configuration with single species."""
        species = PassiveScalarConfig(name="dye", kappa=0.01)
        scalars = ScalarsConfig(enabled=True, species=[species])
        assert scalars.enabled
        assert len(scalars.species) == 1
        assert scalars.species[0].name == "dye"

    def test_multiple_species(self):
        """Test configuration with multiple species."""
        species1 = PassiveScalarConfig(name="temperature", kappa=0.02)
        species2 = PassiveScalarConfig(name="salinity", kappa=0.01)
        scalars = ScalarsConfig(enabled=True, species=[species1, species2])
        assert len(scalars.species) == 2
        assert scalars.species[0].name == "temperature"
        assert scalars.species[1].name == "salinity"

    def test_duplicate_names(self):
        """Test that duplicate species names raise error."""
        species1 = PassiveScalarConfig(name="tracer", kappa=0.01)
        species2 = PassiveScalarConfig(name="tracer", kappa=0.02)

        with pytest.raises(ValueError, match="Duplicate scalar names"):
            ScalarsConfig(enabled=True, species=[species1, species2])


class TestScalarConfigIntegration:
    """Test integration of scalar configuration with RunConfig."""

    def test_runconfig_without_scalars(self):
        """Test RunConfig works without scalars."""
        config = RunConfig(grid=GridConfig(N=128, L=6.28), solver=SolverConfig(alpha=1.0))
        assert config.scalars is None

    def test_runconfig_with_scalars(self):
        """Test RunConfig with scalars configuration."""
        scalars = ScalarsConfig(
            enabled=True, species=[PassiveScalarConfig(name="tracer", kappa=0.01)]
        )
        config = RunConfig(grid=GridConfig(N=128), solver=SolverConfig(alpha=1.0), scalars=scalars)
        assert config.scalars is not None
        assert config.scalars.enabled
        assert len(config.scalars.species) == 1

    def test_yaml_roundtrip(self):
        """Test saving and loading scalar config from YAML."""
        # Create config with scalars
        config = RunConfig(
            grid=GridConfig(N=256),
            solver=SolverConfig(alpha=1.5),
            scalars=ScalarsConfig(
                enabled=True,
                species=[
                    PassiveScalarConfig(
                        name="temperature",
                        kappa=0.01,
                        source=ScalarSourceConfig(
                            type="localized", parameters={"x0": 3.14, "y0": 3.14, "sigma": 0.5}
                        ),
                        initial_condition="gaussian",
                        initial_params={"width": 1.0},
                    ),
                    PassiveScalarConfig(name="dye", kappa=0.001),
                ],
            ),
        )

        # Save to YAML and reload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name

        try:
            # Load and verify
            loaded = RunConfig.from_yaml(temp_path)

            assert loaded.scalars is not None
            assert loaded.scalars.enabled
            assert len(loaded.scalars.species) == 2

            # Check first species
            temp_scalar = loaded.scalars.species[0]
            assert temp_scalar.name == "temperature"
            assert temp_scalar.kappa == 0.01
            assert temp_scalar.source.type == "localized"
            assert temp_scalar.source.parameters["x0"] == 3.14
            assert temp_scalar.initial_condition == "gaussian"

            # Check second species
            dye_scalar = loaded.scalars.species[1]
            assert dye_scalar.name == "dye"
            assert dye_scalar.kappa == 0.001
            assert dye_scalar.source is None

        finally:
            Path(temp_path).unlink()

    def test_from_dict_with_scalars(self):
        """Test from_dict parsing with scalar configuration."""
        data = {
            "grid": {"N": 128, "L": 6.28},
            "solver": {"alpha": 1.0},
            "scalars": {
                "enabled": True,
                "species": [
                    {
                        "name": "tracer",
                        "kappa": 0.01,
                        "source": {"type": "exponential", "parameters": {"rate": 0.1}},
                        "initial_condition": "random",
                        "initial_params": {"seed": 42},
                    }
                ],
            },
        }

        config = RunConfig.from_dict(data)

        assert config.scalars is not None
        assert config.scalars.enabled
        assert len(config.scalars.species) == 1
        assert config.scalars.species[0].name == "tracer"
        assert config.scalars.species[0].source.type == "exponential"
        assert config.scalars.species[0].source.parameters["rate"] == 0.1

    def test_from_dict_disabled_scalars(self):
        """Test from_dict with disabled scalars."""
        data = {"grid": {"N": 128}, "solver": {"alpha": 1.0}, "scalars": {"enabled": False}}

        config = RunConfig.from_dict(data)
        assert config.scalars is None

    def test_from_dict_no_scalars(self):
        """Test from_dict without scalars section."""
        data = {"grid": {"N": 128}, "solver": {"alpha": 1.0}}

        config = RunConfig.from_dict(data)
        assert config.scalars is None

    def test_complex_scalar_yaml(self):
        """Test complex scalar configuration in YAML format."""
        yaml_content = """
grid:
  N: 256
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-16
    p: 8

scalars:
  enabled: true
  species:
    - name: temperature
      kappa: 0.01
      source:
        type: localized
        parameters:
          x0: 3.14159
          y0: 3.14159
          sigma: 0.5
          amplitude: 1.0
      initial_condition: gaussian
      initial_params:
        center: [3.14159, 3.14159]
        width: 1.0
        
    - name: salinity
      kappa: 0.005
      initial_condition: uniform
      initial_params:
        value: 35.0
        
    - name: dye
      kappa: 0.001
      source:
        type: chemical
        parameters:
          rate: 0.1
          threshold: 0.01
      initial_condition: random
      initial_params:
        seed: 42
        amplitude: 0.1

simulation:
  t_end: 100.0
  output_interval: 1.0
"""

        # Write to temp file and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = RunConfig.from_yaml(temp_path)

            # Verify scalar configuration
            assert config.scalars.enabled
            assert len(config.scalars.species) == 3

            # Check temperature
            temp = config.scalars.species[0]
            assert temp.name == "temperature"
            assert temp.kappa == 0.01
            assert temp.source.type == "localized"
            assert temp.source.parameters["sigma"] == 0.5

            # Check salinity
            salt = config.scalars.species[1]
            assert salt.name == "salinity"
            assert salt.kappa == 0.005
            assert salt.source is None
            assert salt.initial_condition == "uniform"

            # Check dye
            dye = config.scalars.species[2]
            assert dye.name == "dye"
            assert dye.source.type == "chemical"
            assert dye.source.parameters["rate"] == 0.1

        finally:
            Path(temp_path).unlink()

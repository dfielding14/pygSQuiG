"""Tests for the configuration system."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from pygsquig.io import (
    RunConfig,
    GridConfig,
    SolverConfig,
    ForcingConfig,
    OutputConfig,
    SimulationConfig,
    InitialConditionConfig,
    DissipationConfig,
    DampingConfig,
    TimeIntegrationConfig,
    load_config,
)


class TestGridConfig:
    """Test GridConfig dataclass."""

    def test_default_values(self):
        """Test default grid configuration."""
        config = GridConfig()
        assert config.N == 256
        assert config.L == 2 * np.pi

    def test_custom_values(self):
        """Test custom grid configuration."""
        config = GridConfig(N=512, L=10.0)
        assert config.N == 512
        assert config.L == 10.0

    def test_validation(self):
        """Test grid configuration validation."""
        # N must be positive even
        with pytest.raises(ValueError, match="positive even integer"):
            GridConfig(N=0)
        with pytest.raises(ValueError, match="positive even integer"):
            GridConfig(N=-10)
        with pytest.raises(ValueError, match="positive even integer"):
            GridConfig(N=255)

        # L must be positive
        with pytest.raises(ValueError, match="L must be positive"):
            GridConfig(L=0)
        with pytest.raises(ValueError, match="L must be positive"):
            GridConfig(L=-1.0)


class TestDissipationConfig:
    """Test DissipationConfig dataclass."""

    def test_default_values(self):
        """Test default dissipation configuration."""
        config = DissipationConfig()
        assert config.type == "hyperviscosity"
        assert config.nu_p == 1.0e-16
        assert config.p == 8

    def test_validation(self):
        """Test dissipation configuration validation."""
        # Invalid type
        with pytest.raises(ValueError, match="Unknown dissipation type"):
            DissipationConfig(type="invalid")

        # nu_p must be non-negative
        with pytest.raises(ValueError, match="nu_p must be non-negative"):
            DissipationConfig(nu_p=-1.0)

        # p must be 2, 4, or 8
        with pytest.raises(ValueError, match="p must be 2, 4, or 8"):
            DissipationConfig(p=3)


class TestTimeIntegrationConfig:
    """Test TimeIntegrationConfig dataclass."""

    def test_default_values(self):
        """Test default time integration configuration."""
        config = TimeIntegrationConfig()
        assert config.method == "RK4"
        assert config.dt == 0.001
        assert config.adaptive_cfl is True
        assert config.cfl_safety == 0.8
        assert config.dt_max is None

    def test_validation(self):
        """Test time integration configuration validation."""
        # Invalid method
        with pytest.raises(ValueError, match="Unknown time integration method"):
            TimeIntegrationConfig(method="RK2")

        # dt must be positive
        with pytest.raises(ValueError, match="dt must be positive"):
            TimeIntegrationConfig(dt=0)

        # cfl_safety must be in (0, 1)
        with pytest.raises(ValueError, match="cfl_safety must be in"):
            TimeIntegrationConfig(cfl_safety=0)
        with pytest.raises(ValueError, match="cfl_safety must be in"):
            TimeIntegrationConfig(cfl_safety=1.5)


class TestSolverConfig:
    """Test SolverConfig dataclass."""

    def test_basic_config(self):
        """Test basic solver configuration."""
        config = SolverConfig(alpha=1.0)
        assert config.alpha == 1.0
        assert config.dissipation.type == "hyperviscosity"
        assert config.damping is None
        assert config.time_integration.method == "RK4"

    def test_validation(self):
        """Test solver configuration validation."""
        # alpha must be in [-2, 2]
        with pytest.raises(ValueError, match="alpha must be in"):
            SolverConfig(alpha=-2.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            SolverConfig(alpha=2.5)


class TestForcingConfig:
    """Test ForcingConfig dataclass."""

    def test_default_values(self):
        """Test default forcing configuration."""
        config = ForcingConfig()
        assert config.type == "ring"
        assert config.kf == 20.0
        assert config.dk == 1.0
        assert config.epsilon == 0.1
        assert config.tau_f == 0.0
        assert config.seed is None

    def test_validation(self):
        """Test forcing configuration validation."""
        # Invalid type
        with pytest.raises(ValueError, match="Unknown forcing type"):
            ForcingConfig(type="invalid")

        # kf must be positive
        with pytest.raises(ValueError, match="kf must be positive"):
            ForcingConfig(kf=0)

        # dk must be positive
        with pytest.raises(ValueError, match="dk must be positive"):
            ForcingConfig(dk=0)

        # epsilon must be non-negative
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            ForcingConfig(epsilon=-0.1)


class TestOutputConfig:
    """Test OutputConfig dataclass."""

    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        assert config.fields == ["theta"]
        assert config.diagnostics == ["energy_spectrum", "scalar_flux"]
        assert config.save_every_n_steps is None
        assert config.compress is True

    def test_validation(self):
        """Test output configuration validation."""
        # Invalid fields
        with pytest.raises(ValueError, match="Unknown fields"):
            OutputConfig(fields=["theta", "invalid_field"])

        # Invalid diagnostics
        with pytest.raises(ValueError, match="Unknown diagnostics"):
            OutputConfig(diagnostics=["invalid_diagnostic"])


class TestSimulationConfig:
    """Test SimulationConfig dataclass."""

    def test_basic_config(self):
        """Test basic simulation configuration."""
        config = SimulationConfig(t_end=50.0)
        assert config.t_end == 50.0
        assert config.output_interval == 1.0
        assert config.checkpoint_interval == 10.0
        assert config.wall_time_limit is None
        assert config.log_interval == 0.1

    def test_validation(self):
        """Test simulation configuration validation."""
        # t_end must be positive
        with pytest.raises(ValueError, match="t_end must be positive"):
            SimulationConfig(t_end=0)

        # output_interval must be positive
        with pytest.raises(ValueError, match="output_interval must be positive"):
            SimulationConfig(t_end=10.0, output_interval=0)


class TestRunConfig:
    """Test RunConfig dataclass and YAML I/O."""

    def test_minimal_config(self):
        """Test minimal run configuration."""
        config = RunConfig(grid=GridConfig(N=128), solver=SolverConfig(alpha=1.0))
        assert config.grid.N == 128
        assert config.solver.alpha == 1.0
        assert config.forcing is None
        assert config.output.fields == ["theta"]
        assert config.simulation.t_end == 100.0

    def test_full_config(self):
        """Test full run configuration."""
        config = RunConfig(
            grid=GridConfig(N=512, L=10.0),
            solver=SolverConfig(
                alpha=1.5,
                dissipation=DissipationConfig(nu_p=1e-8, p=4),
                damping=DampingConfig(mu=0.05),
                time_integration=TimeIntegrationConfig(dt=0.0001),
            ),
            forcing=ForcingConfig(kf=30.0, epsilon=0.2),
            output=OutputConfig(fields=["theta", "vorticity"]),
            simulation=SimulationConfig(t_end=50.0, output_interval=0.5),
            initial_condition=InitialConditionConfig(type="random", seed=42),
        )

        assert config.grid.N == 512
        assert config.solver.alpha == 1.5
        assert config.solver.dissipation.p == 4
        assert config.solver.damping.mu == 0.05
        assert config.forcing.kf == 30.0
        assert config.output.fields == ["theta", "vorticity"]
        assert config.simulation.t_end == 50.0
        assert config.initial_condition.seed == 42

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "grid": {"N": 256, "L": 6.283185307179586},
            "solver": {
                "alpha": 1.0,
                "dissipation": {"type": "hyperviscosity", "nu_p": 1e-16, "p": 8},
            },
            "forcing": {"type": "ring", "kf": 20.0, "epsilon": 0.1},
            "simulation": {"t_end": 100.0},
        }

        config = RunConfig.from_dict(data)
        assert config.grid.N == 256
        assert config.solver.alpha == 1.0
        assert config.forcing.kf == 20.0
        assert config.simulation.t_end == 100.0

    def test_yaml_roundtrip(self):
        """Test saving and loading configuration to/from YAML."""
        config = RunConfig(
            grid=GridConfig(N=512),
            solver=SolverConfig(alpha=1.0),
            forcing=ForcingConfig(kf=25.0),
            simulation=SimulationConfig(t_end=50.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yml"

            # Save to YAML
            config.to_yaml(yaml_path)

            # Load from YAML
            loaded_config = RunConfig.from_yaml(yaml_path)

            # Check values match
            assert loaded_config.grid.N == config.grid.N
            assert loaded_config.solver.alpha == config.solver.alpha
            assert loaded_config.forcing.kf == config.forcing.kf
            assert loaded_config.simulation.t_end == config.simulation.t_end

    def test_yaml_with_missing_sections(self):
        """Test loading YAML with missing optional sections."""
        yaml_content = """
grid:
  N: 128
  L: 6.283185307179586
  
solver:
  alpha: 1.0
  
simulation:
  t_end: 10.0
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "minimal.yml"
            with open(yaml_path, "w") as f:
                f.write(yaml_content)

            config = load_config(yaml_path)
            assert config.grid.N == 128
            assert config.solver.alpha == 1.0
            assert config.forcing is None  # Optional
            assert config.output.fields == ["theta"]  # Default
            assert config.simulation.t_end == 10.0

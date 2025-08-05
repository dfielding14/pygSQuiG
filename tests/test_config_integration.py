"""Test integration of simplified config with existing code."""

import tempfile
from pathlib import Path

import pytest

from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing.ring_forcing import RingForcing
from pygsquig.io import Config, adapt_config


class TestConfigIntegration:
    """Test that simplified config works with existing codebase."""

    def test_with_grid_creation(self):
        """Test config works with grid creation."""
        config = Config({"grid": {"N": 64, "L": 10.0}})
        adapted = adapt_config(config)

        # Use with make_grid
        grid = make_grid(adapted.grid.N, adapted.grid.L)

        assert grid.N == 64
        assert grid.L == 10.0

    def test_with_solver_creation(self):
        """Test config works with solver creation."""
        config = Config({"grid": {"N": 32}, "solver": {"alpha": 0.5, "nu_p": 1e-10, "p": 4}})
        adapted = adapt_config(config)

        grid = make_grid(adapted.grid.N, adapted.grid.L)
        solver = gSQGSolver(
            grid,
            alpha=adapted.solver.alpha,
            nu_p=adapted.solver.dissipation.nu_p,
            p=adapted.solver.dissipation.p,
        )

        assert solver.alpha == 0.5
        assert solver.nu_p == 1e-10
        assert solver.p == 4

    def test_with_forcing_creation(self):
        """Test config works with forcing creation."""
        config = Config({"forcing": {"kf": 25.0, "dk": 2.0, "epsilon": 0.2, "tau_f": 0.5}})
        adapted = adapt_config(config)

        assert adapted.forcing is not None

        forcing = RingForcing(
            kf=adapted.forcing.kf,
            dk=adapted.forcing.dk,
            epsilon=adapted.forcing.epsilon,
            tau_f=adapted.forcing.tau_f,
        )

        assert forcing.kf == 25.0
        assert forcing.epsilon == 0.2

    def test_yaml_config_with_run_patterns(self):
        """Test YAML config works with patterns from run.py."""
        yaml_content = """
grid:
  N: 128
  L: 6.283185307179586

solver:
  alpha: 1.0
  dissipation:
    nu_p: 1.0e-14
    p: 8

forcing:
  type: ring
  kf: 30.0
  dk: 3.0
  epsilon: 0.5

time_integration:
  method: RK4
  adaptive: true
  cfl_safety: 0.8

simulation:
  t_end: 10.0
  output_interval: 1.0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            # Load with simple config
            config = Config.from_yaml(temp_path)
            adapted = adapt_config(config)

            # Patterns from run.py
            grid = make_grid(adapted.grid.N, adapted.grid.L)

            solver = gSQGSolver(
                grid,
                alpha=adapted.solver.alpha,
                nu_p=adapted.solver.dissipation.nu_p,
                p=adapted.solver.dissipation.p,
            )

            if adapted.forcing:
                forcing = RingForcing(
                    kf=adapted.forcing.kf,
                    dk=adapted.forcing.dk,
                    epsilon=adapted.forcing.epsilon,
                    tau_f=adapted.forcing.tau_f or 0.0,
                )
                assert forcing.epsilon == 0.5

            # Check time integration settings
            assert adapted.solver.time_integration.method == "RK4"
            assert adapted.solver.time_integration.adaptive_cfl is True
            assert adapted.solver.time_integration.cfl_safety == 0.8

            # Check simulation settings
            assert adapted.simulation.t_end == 10.0
            assert adapted.simulation.output_interval == 1.0

        finally:
            temp_path.unlink()

    def test_minimal_config(self):
        """Test minimal config with defaults."""
        config = Config({"grid": {"N": 64}, "solver": {"alpha": 1.0}})
        adapted = adapt_config(config)

        # Should have sensible defaults
        assert adapted.grid.N == 64
        assert adapted.grid.L > 0
        assert adapted.solver.alpha == 1.0
        assert adapted.solver.dissipation.nu_p >= 0
        assert adapted.solver.dissipation.p in [2, 4, 8]
        assert adapted.forcing is None  # No forcing by default

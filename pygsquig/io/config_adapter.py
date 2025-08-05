"""Adapter to make simplified Config work with code expecting dataclass configs.

This module provides compatibility between the simplified configuration system
and existing code that expects the complex dataclass-based configurations.
"""

from typing import Any, Optional
from .simple_config import Config


class ConfigAdapter:
    """Adapter that makes Config look like the old dataclass system."""

    def __init__(self, config: Config, section: str):
        self._config = config
        self._section = section

    def __getattr__(self, name: str) -> Any:
        """Get configuration value from the section."""
        key = f"{self._section}.{name}"
        value = self._config.get(key)

        # Special handling for nested configs
        if name == "dissipation" and self._section == "solver":
            return ConfigAdapter(self._config, "solver")
        elif name == "damping" and self._section == "solver":
            # Return None if damping is not enabled
            if not self._config.get("damping.enabled", False):
                return None
            return ConfigAdapter(self._config, "damping")
        elif name == "time_integration" and self._section == "solver":
            return ConfigAdapter(self._config, "time_integration")

        # Special mappings for compatibility
        if self._section == "solver" and name == "nu_p":
            return self._config.get("solver.nu_p")
        elif self._section == "solver" and name == "p":
            return self._config.get("solver.p")
        elif self._section == "damping" and name == "mu":
            return self._config.get("damping.mu")
        elif self._section == "damping" and name == "kf":
            # Calculate kf from forcing.kf and k_cutoff_factor
            forcing_kf = self._config.get("forcing.kf", 20.0)
            factor = self._config.get("damping.k_cutoff_factor", 0.5)
            return forcing_kf * factor

        return value

    def __repr__(self) -> str:
        return f"ConfigAdapter({self._section})"


def adapt_config(config: Config) -> "RunConfigAdapter":
    """Create an adapter that makes Config compatible with old dataclass system.

    Args:
        config: Simplified Config instance

    Returns:
        Adapter that mimics RunConfig interface
    """
    return RunConfigAdapter(config)


class RunConfigAdapter:
    """Adapter that mimics the RunConfig dataclass interface."""

    def __init__(self, config: Config):
        self._config = config

        # Create section adapters
        self.grid = ConfigAdapter(config, "grid")
        self.solver = SolverConfigAdapter(config)
        self.forcing = self._get_forcing_adapter()
        self.output = ConfigAdapter(config, "output")
        self.simulation = ConfigAdapter(config, "simulation")
        self.initial_condition = ConfigAdapter(config, "initial_condition")

    def _get_forcing_adapter(self):
        """Get forcing adapter if forcing is enabled."""
        # If explicitly disabled, return None
        if self._config.get("forcing.enabled") is False:
            return None

        # Check various ways forcing might be specified as enabled
        if (
            self._config.get("forcing.enabled", False)
            or self._config.get("forcing.epsilon", 0) > 0
            or self._config.get("forcing.type") == "ring"
            or self._config.get("forcing.kf") is not None
        ):
            return ConfigAdapter(self._config, "forcing")
        return None


class SolverConfigAdapter:
    """Special adapter for solver config with nested dissipation/damping."""

    def __init__(self, config: Config):
        self._config = config
        self.alpha = config.get("solver.alpha")

        # Create dissipation config
        self.dissipation = type(
            "DissipationConfig",
            (),
            {
                "nu_p": config.get("solver.nu_p"),
                "p": config.get("solver.p"),
                "type": "hyperviscosity",
            },
        )()

        # Create damping config if enabled
        if config.get("damping.enabled", False) or config.get("damping.mu", 0) > 0:
            forcing_kf = config.get("forcing.kf", 20.0)
            factor = config.get("damping.k_cutoff_factor", 0.5)
            self.damping = type(
                "DampingConfig",
                (),
                {"mu": config.get("damping.mu"), "kf": forcing_kf * factor, "type": "linear_drag"},
            )()
        else:
            self.damping = None

        # Create time integration config
        self.time_integration = type(
            "TimeIntegrationConfig",
            (),
            {
                "method": config.get("time_integration.method"),
                "dt": config.get("time_integration.dt"),
                "adaptive_cfl": config.get("time_integration.adaptive"),
                "cfl_safety": config.get("time_integration.cfl_safety"),
                "dt_max": config.get("time_integration.dt_max"),
            },
        )()


# Helper function to convert old RunConfig to new Config
def dataclass_to_config(run_config) -> Config:
    """Convert old dataclass RunConfig to new simplified Config.

    Args:
        run_config: Instance of dataclass-based RunConfig

    Returns:
        Simplified Config instance
    """
    config_dict = {
        "grid": {
            "N": run_config.grid.N,
            "L": run_config.grid.L,
        },
        "solver": {
            "alpha": run_config.solver.alpha,
            "nu_p": run_config.solver.dissipation.nu_p,
            "p": run_config.solver.dissipation.p,
        },
        "time_integration": {
            "method": run_config.solver.time_integration.method,
            "dt": run_config.solver.time_integration.dt,
            "adaptive": run_config.solver.time_integration.adaptive_cfl,
            "cfl_safety": run_config.solver.time_integration.cfl_safety,
            "dt_max": run_config.solver.time_integration.dt_max,
        },
        "simulation": {
            "t_end": run_config.simulation.t_end,
            "output_interval": run_config.simulation.output_interval,
            "checkpoint_interval": run_config.simulation.checkpoint_interval,
            "log_interval": run_config.simulation.log_interval,
        },
        "output": {
            "fields": run_config.output.fields,
            "diagnostics": run_config.output.diagnostics,
            "compress": run_config.output.compress,
        },
        "initial_condition": {
            "type": run_config.initial_condition.type,
            "amplitude": run_config.initial_condition.amplitude,
            "seed": run_config.initial_condition.seed,
            "checkpoint_path": run_config.initial_condition.checkpoint_path,
        },
    }

    # Handle optional forcing
    if run_config.forcing:
        config_dict["forcing"] = {
            "enabled": True,
            "kf": run_config.forcing.kf,
            "dk": run_config.forcing.dk,
            "epsilon": run_config.forcing.epsilon,
            "tau_f": run_config.forcing.tau_f,
            "seed": run_config.forcing.seed,
        }

    # Handle optional damping
    if run_config.solver.damping:
        config_dict["damping"] = {
            "enabled": True,
            "mu": run_config.solver.damping.mu,
            "k_cutoff_factor": run_config.solver.damping.k_cutoff_factor,
        }

    return Config(config_dict)

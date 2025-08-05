"""Simplified configuration system for pygSQuiG simulations.

This module provides a simpler alternative to the complex dataclass-based
configuration system, using a single Config class with schema validation.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import numpy as np


# Default configuration template
DEFAULT_CONFIG = {
    "grid": {
        "N": 256,
        "L": 2 * np.pi,
    },
    "solver": {
        "alpha": 1.0,
        "nu_p": 1.0e-16,
        "p": 8,
    },
    "forcing": {
        "enabled": False,
        "kf": 20.0,
        "dk": 1.0,
        "epsilon": 0.1,
        "tau_f": 0.0,
        "seed": None,
    },
    "damping": {
        "enabled": False,
        "mu": 0.1,
        "k_cutoff_factor": 0.5,
    },
    "time_integration": {
        "method": "RK4",
        "dt": 0.001,
        "adaptive": True,
        "cfl_safety": 0.8,
        "dt_max": None,
    },
    "simulation": {
        "t_end": 100.0,
        "output_interval": 1.0,
        "checkpoint_interval": 10.0,
        "log_interval": 0.1,
    },
    "output": {
        "fields": ["theta"],
        "diagnostics": ["energy", "enstrophy"],
        "compress": True,
    },
    "initial_condition": {
        "type": "random",
        "amplitude": 1.0,
        "seed": None,
        "checkpoint_path": None,
    },
}


# Validation rules
VALIDATION_RULES = {
    "grid.N": lambda x: x > 0 and x % 2 == 0,
    "grid.L": lambda x: x > 0,
    "solver.alpha": lambda x: -2 <= x <= 2,
    "solver.nu_p": lambda x: x >= 0,
    "solver.p": lambda x: x in [2, 4, 8],
    "forcing.kf": lambda x: x > 0,
    "forcing.dk": lambda x: x > 0,
    "forcing.epsilon": lambda x: x >= 0,
    "forcing.tau_f": lambda x: x >= 0,
    "damping.mu": lambda x: x >= 0,
    "damping.k_cutoff_factor": lambda x: x > 0,
    "time_integration.method": lambda x: x in ["RK4", "SSP-RK3"],
    "time_integration.dt": lambda x: x > 0,
    "time_integration.cfl_safety": lambda x: 0 < x < 1,
    "time_integration.dt_max": lambda x: x is None or x > 0,
    "simulation.t_end": lambda x: x > 0,
    "simulation.output_interval": lambda x: x > 0,
    "simulation.checkpoint_interval": lambda x: x > 0,
    "simulation.log_interval": lambda x: x > 0,
    "initial_condition.type": lambda x: x in ["random", "checkpoint", "function"],
    "initial_condition.amplitude": lambda x: x > 0,
}


class Config:
    """Simplified configuration class for pygSQuiG simulations.

    This class provides a simpler interface than the complex dataclass system,
    while maintaining the same functionality and YAML structure.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with optional dictionary.

        Args:
            config_dict: Configuration dictionary (uses defaults if None)
        """
        # Start with defaults
        self._config = self._deep_copy_dict(DEFAULT_CONFIG)

        # Update with provided config
        if config_dict:
            self._deep_update(self._config, config_dict)

        # Auto-enable forcing if forcing parameters are provided
        if "forcing" in (config_dict or {}) and not (
            "enabled" in (config_dict or {}).get("forcing", {})
        ):
            # If forcing params provided but enabled not specified, auto-enable
            if (
                self._config["forcing"].get("epsilon", 0) > 0
                or self._config["forcing"].get("type") == "ring"
            ):
                self._config["forcing"]["enabled"] = True

        # Auto-enable damping if damping parameters are provided
        if "damping" in (config_dict or {}) and not (
            "enabled" in (config_dict or {}).get("damping", {})
        ):
            if self._config["damping"].get("mu", 0) > 0:
                self._config["damping"]["enabled"] = True

        # Validate
        self._validate()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'grid.N' or 'solver.alpha')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'grid.N' or 'solver.alpha')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

        # Validate this specific key
        if key in VALIDATION_RULES:
            if not VALIDATION_RULES[key](value):
                raise ValueError(f"Invalid value for {key}: {value}")

    def _validate(self) -> None:
        """Validate entire configuration."""
        errors = []

        for key, rule in VALIDATION_RULES.items():
            value = self.get(key)
            if value is not None and not rule(value):
                errors.append(f"Invalid {key}: {value}")

        # Special validation rules
        if self.get("initial_condition.type") == "checkpoint":
            if not self.get("initial_condition.checkpoint_path"):
                errors.append("checkpoint_path required when type='checkpoint'")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))

    def _deep_update(self, base: dict, update: dict) -> None:
        """Recursively update base dictionary with update dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _deep_copy_dict(self, d: dict) -> dict:
        """Create a deep copy of a dictionary."""
        import copy

        return copy.deepcopy(d)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to top-level config sections."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


# Backward compatibility function
def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Config instance
    """
    return Config.from_yaml(path)

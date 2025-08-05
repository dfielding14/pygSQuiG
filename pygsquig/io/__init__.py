"""I/O module for pygSQuiG.

This module provides configuration management and HDF5 I/O functionality
for pygSQuiG simulations.
"""

from .config import (
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

from .simple_config import Config
from .config_adapter import adapt_config

from .hdf5_io import (
    save_checkpoint,
    load_checkpoint,
    save_output,
    load_output,
    save_diagnostics,
    load_diagnostics,
)

__all__ = [
    # Original configuration system
    "RunConfig",
    "GridConfig",
    "SolverConfig",
    "ForcingConfig",
    "OutputConfig",
    "SimulationConfig",
    "InitialConditionConfig",
    "DissipationConfig",
    "DampingConfig",
    "TimeIntegrationConfig",
    "load_config",
    # Simplified configuration system
    "Config",
    "adapt_config",
    # HDF5 I/O
    "save_checkpoint",
    "load_checkpoint",
    "save_output",
    "load_output",
    "save_diagnostics",
    "load_diagnostics",
]

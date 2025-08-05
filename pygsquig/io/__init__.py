"""I/O module for pygSQuiG.

This module provides configuration management and HDF5 I/O functionality
for pygSQuiG simulations.
"""

from .config import (
    DampingConfig,
    DissipationConfig,
    ForcingConfig,
    GridConfig,
    InitialConditionConfig,
    OutputConfig,
    RunConfig,
    SimulationConfig,
    SolverConfig,
    TimeIntegrationConfig,
    load_config,
)
from .config_adapter import adapt_config
from .hdf5_io import (
    load_checkpoint,
    load_diagnostics,
    load_output,
    save_checkpoint,
    save_diagnostics,
    save_output,
)
from .simple_config import Config

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

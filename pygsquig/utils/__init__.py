"""Utilities module for pygSQuiG.

This module provides diagnostics, logging, and other utilities
for pygSQuiG simulations.
"""

from .diagnostics import (
    compute_energy_flux,
    compute_energy_spectrum,
    compute_enstrophy,
    compute_palinstrophy,
    compute_scalar_flux,
    compute_total_energy,
)
from .logging import (
    ProgressBar,
    SimulationLogger,
    get_logger,
    setup_logging,
)

__all__ = [
    # Diagnostics
    "compute_energy_spectrum",
    "compute_scalar_flux",
    "compute_enstrophy",
    "compute_energy_flux",
    "compute_total_energy",
    "compute_palinstrophy",
    # Logging
    "SimulationLogger",
    "ProgressBar",
    "setup_logging",
    "get_logger",
]

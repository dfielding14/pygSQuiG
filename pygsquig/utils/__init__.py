"""Utilities module for pygSQuiG.

This module provides diagnostics, logging, and other utilities
for pygSQuiG simulations.
"""

from .diagnostics import (
    compute_energy_spectrum,
    compute_scalar_flux,
    compute_enstrophy,
    compute_energy_flux,
    compute_total_energy,
    compute_palinstrophy,
)

from .logging import (
    SimulationLogger,
    ProgressBar,
    setup_logging,
    get_logger,
)

# Re-export plotting functions from plots package for backward compatibility
from ..plots import (
    PlotStyle,
    plot_field_slice,
    plot_vorticity,
    plot_velocity_fields,
    plot_energy_spectrum_with_analysis,
    plot_diagnostic_summary,
    plot_time_series_multiplot,
    create_field_animation,
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
    # Plotting
    "PlotStyle",
    "plot_field_slice",
    "plot_vorticity",
    "plot_velocity_fields",
    "plot_energy_spectrum_with_analysis",
    "plot_diagnostic_summary",
    "plot_time_series_multiplot",
    "create_field_animation",
]

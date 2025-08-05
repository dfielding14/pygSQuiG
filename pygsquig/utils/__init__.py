"""Utilities module for pygSQuiG.

This module provides diagnostics, logging, and other utilities
for pygSQuiG simulations.
"""

# Re-export plotting functions from plots package for backward compatibility
from ..plots import (
    PlotStyle,
    create_field_animation,
    plot_diagnostic_summary,
    plot_energy_spectrum_with_analysis,
    plot_field_slice,
    plot_time_series_multiplot,
    plot_velocity_fields,
    plot_vorticity,
)
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

"""
Plotting utilities for pygSQuiG simulations.

This package provides comprehensive visualization tools split into modules:
- style: Plot styling configuration
- fields: Field visualization functions
- spectra: Spectrum analysis plots
- timeseries: Time series plotting
- animations: Animation creation
"""

# Import all functions for backward compatibility
from .style import PlotStyle

from .fields import (
    plot_field_slice,
    plot_vorticity,
    plot_velocity_fields,
    plot_diagnostic_summary
)

from .spectra import (
    plot_energy_spectrum_with_analysis,
    plot_spectrum_evolution,
    plot_compensated_spectrum
)

from .timeseries import (
    plot_time_series_multiplot,
    plot_energy_balance,
    plot_conservation_check,
    plot_regime_evolution
)

from .animations import (
    create_field_animation,
    create_vorticity_animation,
    create_spectrum_animation
)

__all__ = [
    # Style
    'PlotStyle',
    
    # Fields
    'plot_field_slice',
    'plot_vorticity',
    'plot_velocity_fields',
    'plot_diagnostic_summary',
    
    # Spectra
    'plot_energy_spectrum_with_analysis',
    'plot_spectrum_evolution',
    'plot_compensated_spectrum',
    
    # Time series
    'plot_time_series_multiplot',
    'plot_energy_balance',
    'plot_conservation_check',
    'plot_regime_evolution',
    
    # Animations
    'create_field_animation',
    'create_vorticity_animation',
    'create_spectrum_animation',
]
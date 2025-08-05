"""
Passive scalar evolution module for pygSQuiG.

This module provides functionality for evolving passive scalars advected by
gSQG velocity fields with user-defined source terms.
"""

from .passive_scalar import PassiveScalarEvolver, MultiSpeciesEvolver
from .state import PassiveScalarState, MultiScalarState
from .source_terms import (
    SourceTerm,
    ExponentialGrowth,
    LocalizedSource,
    ChemicalReaction,
    TimePeriodicSource
)
from .diagnostics import (
    compute_scalar_variance_spectrum,
    compute_scalar_flux,
    compute_scalar_dissipation,
    compute_mixing_efficiency
)

__all__ = [
    # Core evolver
    'PassiveScalarEvolver',
    'MultiSpeciesEvolver',
    
    # State management
    'PassiveScalarState',
    'MultiScalarState',
    
    # Source terms
    'SourceTerm',
    'ExponentialGrowth',
    'LocalizedSource',
    'ChemicalReaction',
    'TimePeriodicSource',
    
    # Diagnostics
    'compute_scalar_variance_spectrum',
    'compute_scalar_flux',
    'compute_scalar_dissipation',
    'compute_mixing_efficiency'
]
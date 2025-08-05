"""
Passive scalar evolution module for pygSQuiG.

This module provides functionality for evolving passive scalars advected by
gSQG velocity fields with user-defined source terms.
"""

from .diagnostics import (
    compute_mixing_efficiency,
    compute_scalar_dissipation,
    compute_scalar_flux,
    compute_scalar_variance_spectrum,
)
from .passive_scalar import MultiSpeciesEvolver, PassiveScalarEvolver
from .source_terms import (
    ChemicalReaction,
    ExponentialGrowth,
    LocalizedSource,
    SourceTerm,
    TimePeriodicSource,
)
from .state import MultiScalarState, PassiveScalarState

__all__ = [
    # Core evolver
    "PassiveScalarEvolver",
    "MultiSpeciesEvolver",
    # State management
    "PassiveScalarState",
    "MultiScalarState",
    # Source terms
    "SourceTerm",
    "ExponentialGrowth",
    "LocalizedSource",
    "ChemicalReaction",
    "TimePeriodicSource",
    # Diagnostics
    "compute_scalar_variance_spectrum",
    "compute_scalar_flux",
    "compute_scalar_dissipation",
    "compute_mixing_efficiency",
]

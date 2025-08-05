"""
Custom exceptions for pygSQuiG.

This module defines exception hierarchy for better error handling
and debugging throughout the codebase.
"""


class pygSQuiGError(Exception):
    """Base exception for all pygSQuiG errors."""
    pass


class ConfigurationError(pygSQuiGError):
    """Raised when configuration parameters are invalid."""
    pass


class SimulationError(pygSQuiGError):
    """Raised during simulation execution."""
    pass


class NumericalError(SimulationError):
    """Raised when numerical instability or convergence failure occurs."""
    pass


class IOError(pygSQuiGError):
    """Raised for file I/O errors."""
    pass


class ValidationError(pygSQuiGError):
    """Raised when parameter validation fails."""
    pass


class PassiveScalarError(pygSQuiGError):
    """Base exception for passive scalar specific errors."""
    pass


class SourceTermError(PassiveScalarError):
    """Raised when source term computation fails."""
    pass


class ForcingError(pygSQuiGError):
    """Raised when forcing configuration or computation fails."""
    pass
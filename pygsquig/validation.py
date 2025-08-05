"""
Parameter validation utilities for pygSQuiG.

This module provides decorators and functions for validating
parameters throughout the codebase.
"""

from functools import wraps
from typing import Callable, Union

import jax.numpy as jnp
import numpy as np

from pygsquig.exceptions import ValidationError


def validate_alpha(func: Callable) -> Callable:
    """Validate that alpha parameter is in valid range [-2, 2].

    Args:
        func: Function to decorate

    Returns:
        Decorated function with alpha validation

    Raises:
        ValidationError: If alpha is outside valid range
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find alpha in args or kwargs
        alpha = kwargs.get("alpha")
        if alpha is None and len(args) > 2:
            # Assume alpha is third positional argument
            alpha = args[2]

        if alpha is None:
            raise ValidationError("alpha parameter required")

        if not -2 <= alpha <= 2:
            raise ValidationError(f"alpha must be in [-2, 2], got {alpha}")

        return func(*args, **kwargs)

    return wrapper


def validate_grid_size(func: Callable) -> Callable:
    """Validate grid size is even, positive, and >= 4.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with grid size validation

    Raises:
        ValidationError: If grid size is invalid
    """

    @wraps(func)
    def wrapper(N: int, *args, **kwargs):
        if not isinstance(N, int):
            raise ValidationError(f"Grid size N must be integer, got {type(N).__name__}")

        if N <= 0:
            raise ValidationError(f"Grid size N must be positive, got {N}")

        if N % 2 != 0:
            raise ValidationError(f"Grid size N must be even, got {N}")

        if N < 4:
            raise ValidationError(f"Grid size N must be at least 4, got {N}")

        return func(N, *args, **kwargs)

    return wrapper


def validate_timestep(dt: float, dt_min: float = 1e-10, dt_max: float = 1.0) -> float:
    """Validate timestep is within reasonable bounds.

    Args:
        dt: Timestep to validate
        dt_min: Minimum allowed timestep
        dt_max: Maximum allowed timestep

    Returns:
        Validated timestep

    Raises:
        ValidationError: If timestep is invalid
    """
    if not isinstance(dt, (float, int, np.floating)):
        raise ValidationError(f"Timestep dt must be numeric, got {type(dt).__name__}")

    dt = float(dt)

    if dt <= 0:
        raise ValidationError(f"Timestep dt must be positive, got {dt}")

    if dt < dt_min:
        raise ValidationError(f"Timestep dt={dt} is below minimum {dt_min}")

    if dt > dt_max:
        raise ValidationError(f"Timestep dt={dt} exceeds maximum {dt_max}")

    return dt


def validate_diffusivity(kappa: float, name: str = "kappa") -> float:
    """Validate diffusivity coefficient is non-negative.

    Args:
        kappa: Diffusivity to validate
        name: Parameter name for error messages

    Returns:
        Validated diffusivity

    Raises:
        ValidationError: If diffusivity is invalid
    """
    if not isinstance(kappa, (float, int, np.floating)):
        raise ValidationError(f"{name} must be numeric, got {type(kappa).__name__}")

    kappa = float(kappa)

    if kappa < 0:
        raise ValidationError(f"{name} must be non-negative, got {kappa}")

    return kappa


def validate_array_shape(
    array: Union[np.ndarray, jnp.ndarray], expected_shape: tuple, name: str = "array"
) -> None:
    """Validate array has expected shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        name: Array name for error messages

    Raises:
        ValidationError: If array shape doesn't match
    """
    if not hasattr(array, "shape"):
        raise ValidationError(f"{name} must be an array, got {type(array).__name__}")

    if array.shape != expected_shape:
        raise ValidationError(f"{name} has shape {array.shape}, expected {expected_shape}")


def validate_complex_field(
    field_hat: Union[np.ndarray, jnp.ndarray], name: str = "field_hat"
) -> None:
    """Validate field is complex with proper dtype.

    Args:
        field_hat: Complex field to validate
        name: Field name for error messages

    Raises:
        ValidationError: If field is not complex
    """
    if not hasattr(field_hat, "dtype"):
        raise ValidationError(f"{name} must be an array, got {type(field_hat).__name__}")

    if not np.issubdtype(field_hat.dtype, np.complexfloating):
        raise ValidationError(f"{name} must be complex dtype, got {field_hat.dtype}")

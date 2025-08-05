"""
Source term implementations for passive scalar evolution.

This module provides various source term models including growth/decay,
localized sources, and chemical reactions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pygsquig.core.grid import Grid
from pygsquig.exceptions import SourceTermError
from pygsquig.validation import validate_array_shape


class SourceTerm(ABC):
    """Abstract base class for passive scalar source terms.

    All source terms should implement the __call__ method that
    returns the source in physical space.
    """

    @abstractmethod
    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute source term in physical space.

        Args:
            scalar: Scalar field in physical space
            grid: Grid object
            t: Current time

        Returns:
            Source term in physical space
        """
        pass

    def validate_input(self, scalar: jnp.ndarray, grid: Grid) -> None:
        """Validate input arrays have correct shape.

        Args:
            scalar: Scalar field to validate
            grid: Grid object for shape reference

        Raises:
            SourceTermError: If validation fails
        """
        try:
            validate_array_shape(scalar, (grid.N, grid.N), "scalar")
        except Exception as e:
            raise SourceTermError(f"Invalid input to source term: {e}")


@dataclass
class ExponentialGrowth(SourceTerm):
    """Exponential growth/decay source term: S = λ * θ.

    Attributes:
        rate: Growth rate (positive) or decay rate (negative)
    """

    rate: float

    def __post_init__(self):
        """Validate parameters."""
        if not isinstance(self.rate, (float, int)):
            raise SourceTermError(f"Rate must be numeric, got {type(self.rate).__name__}")

    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute exponential growth/decay source.

        Args:
            scalar: Scalar field in physical space
            grid: Grid object (unused)
            t: Current time (unused)

        Returns:
            Source term S = rate * scalar
        """
        return self.rate * scalar


@dataclass
class LocalizedSource(SourceTerm):
    """Gaussian localized source: S = A * exp(-r²/σ²).

    Attributes:
        amplitude: Source amplitude
        x0: x-coordinate of source center
        y0: y-coordinate of source center
        sigma: Gaussian width
        time_dependent: Optional time modulation function
    """

    amplitude: float
    x0: float
    y0: float
    sigma: float
    time_dependent: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate and precompute parameters."""
        if self.sigma <= 0:
            raise SourceTermError(f"Sigma must be positive, got {self.sigma}")
        self.sigma2 = self.sigma**2

    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute localized Gaussian source.

        Args:
            scalar: Scalar field in physical space (unused)
            grid: Grid object
            t: Current time

        Returns:
            Localized source term
        """
        # Time modulation
        if self.time_dependent is not None:
            amplitude = self.amplitude * self.time_dependent(t)
        else:
            amplitude = self.amplitude

        # Compute Gaussian source
        return self._compute_gaussian(grid.x, grid.y, amplitude)

    @staticmethod
    @jax.jit
    def _compute_gaussian_jit(
        x: jnp.ndarray, y: jnp.ndarray, x0: float, y0: float, sigma2: float, amplitude: float
    ) -> jnp.ndarray:
        """JIT-compiled Gaussian computation.

        Args:
            x: x-coordinate array
            y: y-coordinate array
            x0: x-coordinate of center
            y0: y-coordinate of center
            sigma2: Gaussian width squared
            amplitude: Current amplitude

        Returns:
            Gaussian source field
        """
        # Handle periodic boundaries
        dx = jnp.minimum(jnp.abs(x - x0), jnp.abs(x - x0 + 2 * jnp.pi))
        dx = jnp.minimum(dx, jnp.abs(x - x0 - 2 * jnp.pi))

        dy = jnp.minimum(jnp.abs(y - y0), jnp.abs(y - y0 + 2 * jnp.pi))
        dy = jnp.minimum(dy, jnp.abs(y - y0 - 2 * jnp.pi))

        r2 = dx**2 + dy**2
        return amplitude * jnp.exp(-r2 / sigma2)

    def _compute_gaussian(self, x: jnp.ndarray, y: jnp.ndarray, amplitude: float) -> jnp.ndarray:
        """Compute Gaussian using JIT-compiled function."""
        return self._compute_gaussian_jit(x, y, self.x0, self.y0, self.sigma2, amplitude)


@dataclass
class ChemicalReaction(SourceTerm):
    """Quadratic chemical reaction: S = -k * θ².

    Models second-order chemical decay or reaction.

    Attributes:
        rate: Reaction rate constant (positive for decay)
        threshold: Optional threshold below which reaction stops
    """

    rate: float
    threshold: Optional[float] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.rate < 0:
            raise SourceTermError(f"Reaction rate must be non-negative, got {self.rate}")

    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute chemical reaction source.

        Args:
            scalar: Scalar field in physical space
            grid: Grid object (unused)
            t: Current time (unused)

        Returns:
            Source term S = -rate * scalar²
        """
        source = -self.rate * scalar**2

        # Apply threshold if specified
        if self.threshold is not None:
            source = jnp.where(scalar > self.threshold, source, 0.0)

        return source


@dataclass
class TimePeriodicSource(SourceTerm):
    """Time-periodic forcing: S = A * sin(ωt + φ) * f(x,y).

    Attributes:
        amplitude: Source amplitude
        frequency: Angular frequency (rad/time)
        phase: Phase shift (radians)
        spatial_pattern: Function returning spatial pattern
    """

    amplitude: float
    frequency: float
    phase: float = 0.0
    spatial_pattern: Optional[Callable[[Grid], jnp.ndarray]] = None

    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute time-periodic source.

        Args:
            scalar: Scalar field in physical space (unused)
            grid: Grid object
            t: Current time

        Returns:
            Time-periodic source term
        """
        # Time modulation
        time_factor = self.amplitude * jnp.sin(self.frequency * t + self.phase)

        # Spatial pattern
        if self.spatial_pattern is not None:
            spatial = self.spatial_pattern(grid)
        else:
            # Default: uniform forcing
            spatial = jnp.ones((grid.N, grid.N))

        return time_factor * spatial


class LinearCombination(SourceTerm):
    """Linear combination of multiple source terms.

    Allows combining multiple sources: S = Σ(weight_i * source_i)

    Attributes:
        sources: List of (weight, source) tuples
    """

    def __init__(self, sources: list):
        """Initialize with list of weighted sources.

        Args:
            sources: List of (weight, SourceTerm) tuples
        """
        self.sources = sources

        # Validate
        for weight, source in sources:
            if not isinstance(source, SourceTerm):
                raise SourceTermError(f"Expected SourceTerm, got {type(source).__name__}")

    def __call__(self, scalar: jnp.ndarray, grid: Grid, t: float) -> jnp.ndarray:
        """Compute combined source term.

        Args:
            scalar: Scalar field in physical space
            grid: Grid object
            t: Current time

        Returns:
            Combined source term
        """
        result = jnp.zeros_like(scalar)

        for weight, source in self.sources:
            result += weight * source(scalar, grid, t)

        return result


# Predefined source term factories
def make_heating_source(
    x0: float, y0: float, radius: float = 0.5, power: float = 1.0
) -> LocalizedSource:
    """Create a localized heating source.

    Args:
        x0: x-coordinate of heating center
        y0: y-coordinate of heating center
        radius: Effective radius of heating
        power: Heating power

    Returns:
        LocalizedSource configured for heating
    """
    return LocalizedSource(amplitude=power, x0=x0, y0=y0, sigma=radius / 2.0)  # 2-sigma ~ radius


def make_cooling_source(decay_time: float) -> ExponentialGrowth:
    """Create uniform cooling (exponential decay).

    Args:
        decay_time: Time scale for decay

    Returns:
        ExponentialGrowth configured for cooling
    """
    return ExponentialGrowth(rate=-1.0 / decay_time)


def make_reaction_source(reaction_rate: float, threshold: float = 0.0) -> ChemicalReaction:
    """Create a chemical reaction source.

    Args:
        reaction_rate: Reaction rate constant
        threshold: Concentration threshold for reaction

    Returns:
        ChemicalReaction configured with parameters
    """
    return ChemicalReaction(rate=reaction_rate, threshold=threshold)

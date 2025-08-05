"""
Combined forcing patterns for pygSQuiG.

This module provides utilities for combining multiple forcing patterns
with different weights and combination strategies.
"""

from typing import List, Literal, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array

from pygsquig.core.grid import Grid
from pygsquig.exceptions import ForcingError


class CombinedForcing:
    """Combine multiple forcing patterns.

    This class allows combining deterministic and stochastic forcing
    patterns with specified weights.
    """

    def __init__(
        self,
        forcings: List,
        weights: Optional[List[float]] = None,
        combination_type: Literal["additive", "multiplicative"] = "additive",
    ):
        """Initialize combined forcing.

        Args:
            forcings: List of forcing instances
            weights: Weights for each forcing (normalized internally)
            combination_type: How to combine forcings
        """
        if not forcings:
            raise ForcingError("Must provide at least one forcing")

        self.forcings = forcings

        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(forcings)
        elif len(weights) != len(forcings):
            raise ForcingError("Number of weights must match number of forcings")

        # Normalize weights for additive combination
        if combination_type == "additive":
            total_weight = sum(weights)
            if total_weight <= 0:
                raise ForcingError("Sum of weights must be positive")
            self.weights = [w / total_weight for w in weights]
        else:
            self.weights = weights

        self.combination_type = combination_type

    def __call__(
        self, theta_hat: Array, key: Optional[jax.random.PRNGKey], dt: float, grid: Grid
    ) -> Array:
        """Apply combined forcing.

        Args:
            theta_hat: Current field in spectral space
            key: Random key for stochastic forcings
            dt: Time step
            grid: Grid object

        Returns:
            Combined forcing in spectral space
        """
        if self.combination_type == "additive":
            return self._additive_combination(theta_hat, key, dt, grid)
        else:
            return self._multiplicative_combination(theta_hat, key, dt, grid)

    def _additive_combination(
        self, theta_hat: Array, key: Optional[jax.random.PRNGKey], dt: float, grid: Grid
    ) -> Array:
        """Additive combination of forcings."""
        # Initialize with zeros
        combined_forcing = jnp.zeros_like(theta_hat)

        # Split key if needed
        if key is not None:
            keys = jax.random.split(key, len(self.forcings))
        else:
            keys = [None] * len(self.forcings)

        # Add weighted contributions
        for forcing, weight, subkey in zip(self.forcings, self.weights, keys):
            # Handle both deterministic and stochastic forcings
            if hasattr(forcing, "__call__"):
                # Check if forcing expects a key parameter
                import inspect

                sig = inspect.signature(forcing.__call__)
                if "key" in sig.parameters:
                    contribution = forcing(theta_hat, subkey, dt, grid)
                else:
                    # Deterministic forcing
                    contribution = forcing(theta_hat, dt, grid)
            else:
                raise ForcingError(f"Forcing {forcing} is not callable")

            combined_forcing = combined_forcing + weight * contribution

        return combined_forcing

    def _multiplicative_combination(
        self, theta_hat: Array, key: Optional[jax.random.PRNGKey], dt: float, grid: Grid
    ) -> Array:
        """Multiplicative combination of forcings."""
        # Start with first forcing
        if key is not None:
            keys = jax.random.split(key, len(self.forcings))
        else:
            keys = [None] * len(self.forcings)

        # Get first forcing
        forcing = self.forcings[0]
        import inspect

        sig = inspect.signature(forcing.__call__)
        if "key" in sig.parameters:
            combined_forcing = forcing(theta_hat, keys[0], dt, grid)
        else:
            combined_forcing = forcing(theta_hat, dt, grid)

        # Apply weight
        combined_forcing = self.weights[0] * combined_forcing

        # Multiply by remaining forcings
        for i in range(1, len(self.forcings)):
            forcing = self.forcings[i]
            sig = inspect.signature(forcing.__call__)

            if "key" in sig.parameters:
                contribution = forcing(theta_hat, keys[i], dt, grid)
            else:
                contribution = forcing(theta_hat, dt, grid)

            # Multiplicative combination in physical space
            combined_phys = grid.ifft2(combined_forcing)
            contrib_phys = grid.ifft2(contribution)

            combined_phys = combined_phys * contrib_phys * self.weights[i]
            combined_forcing = grid.fft2(combined_phys)

        return combined_forcing


class TimeModulatedForcing:
    """Wrapper to add time modulation to any forcing pattern."""

    def __init__(
        self,
        base_forcing,
        modulation_type: Literal["oscillatory", "pulsed", "growing", "decaying"] = "oscillatory",
        frequency: float = 1.0,
        growth_rate: float = 0.1,
        duty_cycle: float = 0.5,
    ):
        """Initialize time-modulated forcing.

        Args:
            base_forcing: Base forcing pattern
            modulation_type: Type of time modulation
            frequency: Oscillation/pulse frequency
            growth_rate: Growth/decay rate
            duty_cycle: Fraction of time forcing is on (for pulsed)
        """
        self.base_forcing = base_forcing
        self.modulation_type = modulation_type
        self.frequency = frequency
        self.growth_rate = growth_rate
        self.duty_cycle = duty_cycle
        self._time = 0.0

    def __call__(
        self, theta_hat: Array, key: Optional[jax.random.PRNGKey], dt: float, grid: Grid
    ) -> Array:
        """Apply time-modulated forcing."""
        # Update time
        self._time += dt

        # Get modulation factor
        if self.modulation_type == "oscillatory":
            modulation = jnp.cos(2 * jnp.pi * self.frequency * self._time)
        elif self.modulation_type == "pulsed":
            phase = (self.frequency * self._time) % 1.0
            modulation = 1.0 if phase < self.duty_cycle else 0.0
        elif self.modulation_type == "growing":
            modulation = jnp.exp(self.growth_rate * self._time)
        elif self.modulation_type == "decaying":
            modulation = jnp.exp(-self.growth_rate * self._time)
        else:
            modulation = 1.0

        # Get base forcing
        import inspect

        sig = inspect.signature(self.base_forcing.__call__)
        if "key" in sig.parameters:
            base_forcing = self.base_forcing(theta_hat, key, dt, grid)
        else:
            base_forcing = self.base_forcing(theta_hat, dt, grid)

        return modulation * base_forcing


class MaskForcing:
    """Apply spatial masking to forcing patterns."""

    def __init__(
        self,
        base_forcing,
        mask_type: Literal["annular", "rectangular", "custom"] = "annular",
        inner_radius: float = 0.0,
        outer_radius: float = 1.0,
        x_bounds: tuple = (0.0, 1.0),
        y_bounds: tuple = (0.0, 1.0),
        custom_mask: Optional[Array] = None,
    ):
        """Initialize masked forcing.

        Args:
            base_forcing: Base forcing pattern
            mask_type: Type of spatial mask
            inner_radius: Inner radius for annular mask (fraction of L/2)
            outer_radius: Outer radius for annular mask
            x_bounds: x-bounds for rectangular mask (fractions of L)
            y_bounds: y-bounds for rectangular mask
            custom_mask: Custom mask array
        """
        self.base_forcing = base_forcing
        self.mask_type = mask_type
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.custom_mask = custom_mask
        self._mask_computed = False
        self._mask = None

    def __call__(
        self, theta_hat: Array, key: Optional[jax.random.PRNGKey], dt: float, grid: Grid
    ) -> Array:
        """Apply masked forcing."""
        # Compute mask if needed
        if not self._mask_computed:
            self._compute_mask(grid)

        # Get base forcing
        import inspect

        sig = inspect.signature(self.base_forcing.__call__)
        if "key" in sig.parameters:
            base_forcing = self.base_forcing(theta_hat, key, dt, grid)
        else:
            base_forcing = self.base_forcing(theta_hat, dt, grid)

        # Apply mask in physical space
        forcing_phys = grid.ifft2(base_forcing)
        forcing_phys = forcing_phys * self._mask

        return grid.fft2(forcing_phys)

    def _compute_mask(self, grid: Grid):
        """Compute spatial mask."""
        x, y = grid.x, grid.y
        L = grid.L

        if self.mask_type == "annular":
            # Annular mask centered at domain center
            x_c, y_c = L / 2, L / 2
            r = jnp.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            r_inner = self.inner_radius * L / 2
            r_outer = self.outer_radius * L / 2

            self._mask = (r >= r_inner) & (r <= r_outer)
            self._mask = self._mask.astype(float)

        elif self.mask_type == "rectangular":
            # Rectangular mask
            x_min, x_max = self.x_bounds[0] * L, self.x_bounds[1] * L
            y_min, y_max = self.y_bounds[0] * L, self.y_bounds[1] * L

            self._mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            self._mask = self._mask.astype(float)

        elif self.mask_type == "custom":
            if self.custom_mask is None:
                raise ForcingError("Custom mask not provided")
            self._mask = self.custom_mask

        self._mask_computed = True

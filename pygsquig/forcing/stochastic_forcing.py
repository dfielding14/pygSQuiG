"""
Stochastic forcing patterns for pygSQuiG.

This module implements various random forcing patterns including
white noise, colored noise, and stochastic vortex injection.
"""

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.exceptions import ForcingError


class StochasticForcing(ABC):
    """Abstract base class for stochastic forcing patterns.

    All stochastic forcing inherits from this class and must
    implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid) -> Array:
        """Apply stochastic forcing.

        Args:
            theta_hat: Current theta in spectral space
            key: Random key for stochastic forcing
            dt: Time step
            grid: Grid object

        Returns:
            Forcing increment in spectral space
        """
        pass


class WhiteNoiseForcing(StochasticForcing):
    """White noise forcing in spectral space.

    Adds uncorrelated random perturbations at each wavenumber,
    with optional spectral filtering.
    """

    def __init__(
        self,
        amplitude: float = 0.1,
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
        isotropy: bool = True,
    ):
        """Initialize white noise forcing.

        Args:
            amplitude: Forcing amplitude (energy injection rate)
            k_min: Minimum wavenumber (None for no limit)
            k_max: Maximum wavenumber (None for no limit)
            isotropy: Whether to maintain isotropy
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")

        self.amplitude = amplitude
        self.k_min = k_min
        self.k_max = k_max
        self.isotropy = isotropy

    def __call__(self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid) -> Array:
        """Apply white noise forcing.

        Args:
            theta_hat: Current theta in spectral space
            key: Random key
            dt: Time step
            grid: Grid object

        Returns:
            Forcing increment
        """
        # Split key for real and imaginary parts
        key_real, key_imag = jax.random.split(key, 2)

        # Generate random noise
        # Note: variance scales with sqrt(dt) for proper diffusion
        std = self.amplitude * jnp.sqrt(dt)

        noise_real = jax.random.normal(key_real, theta_hat.shape) * std
        noise_imag = jax.random.normal(key_imag, theta_hat.shape) * std
        noise = noise_real + 1j * noise_imag

        # Apply spectral mask if needed
        if self.k_min is not None or self.k_max is not None:
            mask = self._create_spectral_mask(grid)
            noise = noise * mask

        # Ensure reality condition for physical space
        noise = self._enforce_reality_condition(noise, grid)

        # Normalize to maintain energy injection rate
        if self.isotropy:
            noise = self._normalize_energy_injection(noise, grid, dt)

        # Ensure k=0 mode is zero (no mean injection)
        noise = noise.at[0, 0].set(0.0)

        return noise

    def _create_spectral_mask(self, grid: Grid) -> Array:
        """Create mask for spectral filtering."""
        kx = grid.kx
        ky = grid.ky
        k_squared = kx**2 + ky**2
        k = jnp.sqrt(k_squared)

        mask = jnp.ones_like(k)

        if self.k_min is not None:
            mask = mask * (k >= self.k_min)

        if self.k_max is not None:
            mask = mask * (k <= self.k_max)

        # Zero out k=0 mode
        mask = mask.at[0, 0].set(0)

        return mask

    def _enforce_reality_condition(self, noise: Array, grid: Grid) -> Array:
        """Ensure noise satisfies reality condition in physical space."""
        N = grid.N

        # For real physical fields, we need f(-k) = f*(k)
        # This is automatically satisfied for rfft, but for full FFT we need to enforce it

        # Set k=0 mode to real
        noise = noise.at[0, 0].set(noise[0, 0].real)

        # Set Nyquist modes to real
        noise = noise.at[N // 2, 0].set(noise[N // 2, 0].real)
        noise = noise.at[0, N // 2].set(noise[0, N // 2].real)
        noise = noise.at[N // 2, N // 2].set(noise[N // 2, N // 2].real)

        return noise

    def _normalize_energy_injection(self, noise: Array, grid: Grid, dt: float) -> Array:
        """Normalize to achieve target energy injection rate."""
        # Current energy injection from noise
        energy_injection = jnp.sum(jnp.abs(noise) ** 2).real / grid.N**4

        # Target injection based on amplitude
        target_injection = self.amplitude**2 * dt

        # Scale to match target
        if energy_injection > 0:
            scale = jnp.sqrt(target_injection / energy_injection)
            noise = noise * scale

        return noise


class ColoredNoiseForcing(StochasticForcing):
    """Colored noise forcing with specified power spectrum.

    Generates stochastic forcing with power-law spectrum.
    """

    def __init__(
        self,
        amplitude: float = 0.1,
        spectral_slope: float = -2.0,
        k_peak: float = 10.0,
        k_width: float = 5.0,
    ):
        """Initialize colored noise forcing.

        Args:
            amplitude: Forcing amplitude
            spectral_slope: Power law slope (e.g., -2 for red noise)
            k_peak: Peak wavenumber for forcing
            k_width: Width of forcing band
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")

        self.amplitude = amplitude
        self.spectral_slope = spectral_slope
        self.k_peak = k_peak
        self.k_width = k_width

    def __call__(self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid) -> Array:
        """Apply colored noise forcing."""
        # Split keys
        key_real, key_imag = jax.random.split(key, 2)

        # Generate white noise
        noise_real = jax.random.normal(key_real, theta_hat.shape)
        noise_imag = jax.random.normal(key_imag, theta_hat.shape)
        noise = noise_real + 1j * noise_imag

        # Apply spectral coloring
        kx = grid.kx
        ky = grid.ky
        k_squared = kx**2 + ky**2
        k = jnp.sqrt(k_squared)

        # Power spectrum with peak
        spectrum = jnp.where(
            k > 0,
            (k / self.k_peak) ** self.spectral_slope
            * jnp.exp(-(((k - self.k_peak) / self.k_width) ** 2)),
            0.0,
        )

        # Apply spectrum
        noise = noise * jnp.sqrt(spectrum) * self.amplitude * jnp.sqrt(dt)

        # Ensure reality condition
        noise = self._enforce_reality_condition(noise, grid)

        # Ensure k=0 mode is zero
        noise = noise.at[0, 0].set(0.0)

        return noise

    def _enforce_reality_condition(self, noise: Array, grid: Grid) -> Array:
        """Ensure noise satisfies reality condition."""
        N = grid.N

        # Set k=0 mode to real
        noise = noise.at[0, 0].set(noise[0, 0].real)

        # Set Nyquist modes to real
        noise = noise.at[N // 2, 0].set(noise[N // 2, 0].real)
        noise = noise.at[0, N // 2].set(noise[0, N // 2].real)
        noise = noise.at[N // 2, N // 2].set(noise[N // 2, N // 2].real)

        return noise


class StochasticVortexForcing(StochasticForcing):
    """Random vortex injection forcing.

    Periodically injects coherent vortex structures at random locations.
    """

    def __init__(
        self,
        amplitude: float = 0.1,
        vortex_size: float = 0.1,
        injection_rate: float = 1.0,
        vortex_strength_std: float = 0.2,
    ):
        """Initialize vortex injection forcing.

        Args:
            amplitude: Base vortex amplitude
            vortex_size: Characteristic vortex size (fraction of domain)
            injection_rate: Average number of vortices per unit time
            vortex_strength_std: Standard deviation of vortex strengths
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")
        if vortex_size <= 0 or vortex_size > 1:
            raise ForcingError("Vortex size must be in (0, 1]")
        if injection_rate < 0:
            raise ForcingError("Injection rate must be non-negative")

        self.amplitude = amplitude
        self.vortex_size = vortex_size
        self.injection_rate = injection_rate
        self.vortex_strength_std = vortex_strength_std

    def __call__(self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid) -> Array:
        """Apply vortex injection forcing."""
        # Determine number of vortices to inject
        key_num, key_vortices = jax.random.split(key, 2)
        expected_vortices = self.injection_rate * dt
        n_vortices = jax.random.poisson(key_num, expected_vortices)

        # Generate vortices
        forcing = jnp.zeros_like(theta_hat)

        if n_vortices > 0:
            # Split key for vortex properties
            keys = jax.random.split(key_vortices, 4)

            # Random positions
            x_positions = jax.random.uniform(keys[0], (n_vortices,)) * grid.L
            y_positions = jax.random.uniform(keys[1], (n_vortices,)) * grid.L

            # Random strengths
            strengths = self.amplitude * (
                1 + self.vortex_strength_std * jax.random.normal(keys[2], (n_vortices,))
            )

            # Random signs (cyclonic/anticyclonic)
            signs = 2 * jax.random.bernoulli(keys[3], 0.5, (n_vortices,)) - 1

            # Create vortices in physical space
            x, y = grid.x, grid.y
            radius = self.vortex_size * grid.L

            for i in range(n_vortices):
                # Gaussian vortex profile
                r_squared = (x - x_positions[i]) ** 2 + (y - y_positions[i]) ** 2
                vortex = strengths[i] * signs[i] * jnp.exp(-r_squared / (2 * radius**2))

                # Add to forcing (accumulate in physical space)
                forcing_phys = ifft2(forcing)
                forcing_phys = forcing_phys + vortex
                forcing = fft2(forcing_phys)

        return forcing


class OrnsteinUhlenbeckForcing(StochasticForcing):
    """Ornstein-Uhlenbeck process forcing.

    Generates temporally correlated stochastic forcing with
    exponential decorrelation time.
    """

    def __init__(
        self,
        amplitude: float = 0.1,
        correlation_time: float = 1.0,
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
    ):
        """Initialize OU forcing.

        Args:
            amplitude: Forcing amplitude
            correlation_time: Temporal correlation time
            k_min: Minimum wavenumber
            k_max: Maximum wavenumber
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")
        if correlation_time <= 0:
            raise ForcingError("Correlation time must be positive")

        self.amplitude = amplitude
        self.correlation_time = correlation_time
        self.k_min = k_min
        self.k_max = k_max

        # State for OU process
        self._ou_state = None

    def __call__(self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid) -> Array:
        """Apply OU forcing."""
        # Initialize state if needed
        if self._ou_state is None:
            key_init, key = jax.random.split(key, 2)
            self._ou_state = jax.random.normal(key_init, theta_hat.shape) + 1j * jax.random.normal(
                key_init, theta_hat.shape
            )

        # OU process evolution
        # dX = -X/tau * dt + sqrt(2/tau) * dW
        key_noise = key

        decay = jnp.exp(-dt / self.correlation_time)
        noise_amp = self.amplitude * jnp.sqrt(2 * dt / self.correlation_time * (1 - decay**2))

        # Generate noise
        noise = jax.random.normal(key_noise, theta_hat.shape) + 1j * jax.random.normal(
            key_noise, theta_hat.shape
        )

        # Update OU state
        self._ou_state = decay * self._ou_state + noise_amp * noise

        # Apply spectral mask
        if self.k_min is not None or self.k_max is not None:
            mask = self._create_spectral_mask(grid)
            forcing = self._ou_state * mask
        else:
            forcing = self._ou_state

        # Ensure reality condition
        forcing = self._enforce_reality_condition(forcing, grid)

        # Ensure k=0 mode is zero
        forcing = forcing.at[0, 0].set(0.0)

        return forcing

    def _create_spectral_mask(self, grid: Grid) -> Array:
        """Create spectral mask."""
        kx = grid.kx
        ky = grid.ky
        k_squared = kx**2 + ky**2
        k = jnp.sqrt(k_squared)

        mask = jnp.ones_like(k)

        if self.k_min is not None:
            mask = mask * (k >= self.k_min)

        if self.k_max is not None:
            mask = mask * (k <= self.k_max)

        # Zero out k=0 mode
        mask = mask.at[0, 0].set(0)

        return mask

    def _enforce_reality_condition(self, forcing: Array, grid: Grid) -> Array:
        """Ensure forcing satisfies reality condition."""
        N = grid.N

        # Set k=0 mode to real
        forcing = forcing.at[0, 0].set(forcing[0, 0].real)

        # Set Nyquist modes to real
        forcing = forcing.at[N // 2, 0].set(forcing[N // 2, 0].real)
        forcing = forcing.at[0, N // 2].set(forcing[0, N // 2].real)
        forcing = forcing.at[N // 2, N // 2].set(forcing[N // 2, N // 2].real)

        return forcing


def create_combined_stochastic_forcing(
    forcings: list[StochasticForcing], weights: Optional[list[float]] = None
) -> StochasticForcing:
    """Create a combined stochastic forcing from multiple patterns.

    Args:
        forcings: List of stochastic forcing instances
        weights: Optional weights for each forcing (normalized internally)

    Returns:
        Combined forcing instance
    """
    if not forcings:
        raise ForcingError("Must provide at least one forcing")

    if weights is None:
        weights = [1.0] * len(forcings)
    elif len(weights) != len(forcings):
        raise ForcingError("Number of weights must match number of forcings")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ForcingError("Sum of weights must be positive")

    normalized_weights = [w / total_weight for w in weights]

    class CombinedStochasticForcing(StochasticForcing):
        """Combined stochastic forcing pattern."""

        def __init__(self):
            self.forcings = forcings
            self.weights = normalized_weights

        def __call__(
            self, theta_hat: Array, key: jax.random.PRNGKey, dt: float, grid: Grid
        ) -> Array:
            """Apply combined forcing."""
            # Split key for each forcing
            keys = jax.random.split(key, len(self.forcings))

            # Apply each forcing with its weight
            total_forcing = jnp.zeros_like(theta_hat)

            for forcing, weight, subkey in zip(self.forcings, self.weights, keys):
                contribution = forcing(theta_hat, subkey, dt, grid)
                total_forcing = total_forcing + weight * contribution

            return total_forcing

    return CombinedStochasticForcing()

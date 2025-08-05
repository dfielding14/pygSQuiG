"""
Deterministic forcing patterns for turbulence simulations.

This module provides various deterministic forcing patterns including
Taylor-Green, Kolmogorov flow, and other canonical configurations
used in computational fluid dynamics research.
"""

from typing import Optional, Literal, Union, Callable
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from pygsquig.core.grid import Grid, fft2, ifft2
from pygsquig.validation import validate_diffusivity
from pygsquig.exceptions import ForcingError


class DeterministicForcing(ABC):
    """Abstract base class for deterministic forcing patterns.
    
    All deterministic forcing implementations should inherit from this
    class and implement the __call__ method.
    """
    
    @abstractmethod
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,  # Unused but kept for interface compatibility
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute forcing for one time step.
        
        Args:
            theta_hat: Current state in spectral space
            key: PRNG key (unused for deterministic forcing)
            dt: Time step
            grid: Grid object
            
        Returns:
            Forcing in spectral space F̂
        """
        pass
    
    def get_diagnostics(self, theta_hat: jax.Array, forcing: jax.Array, grid: Grid) -> dict:
        """Compute forcing diagnostics.
        
        Args:
            theta_hat: Current state
            forcing: Current forcing
            grid: Grid object
            
        Returns:
            Dictionary with diagnostic values
        """
        # Energy injection rate
        theta = ifft2(theta_hat).real
        force_phys = ifft2(forcing).real
        injection_rate = jnp.mean(theta * force_phys)
        
        # Total forcing power
        forcing_power = jnp.sum(jnp.abs(forcing)**2) / grid.N**2
        
        return {
            'injection_rate': float(injection_rate),
            'forcing_power': float(forcing_power),
            'forcing_type': self.__class__.__name__
        }


class TaylorGreenForcing(DeterministicForcing):
    """Taylor-Green vortex forcing pattern.
    
    Implements the classic Taylor-Green vortex forcing:
    F = A * sin(k*x) * cos(k*y)
    
    This creates a steady array of counter-rotating vortices.
    
    Attributes:
        amplitude: Forcing amplitude
        k: Wavenumber (number of vortex pairs)
        time_dependent: Whether amplitude varies with time
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        k: int = 2,
        time_dependent: bool = False
    ):
        """Initialize Taylor-Green forcing.
        
        Args:
            amplitude: Base forcing amplitude
            k: Number of vortex pairs in each direction
            time_dependent: If True, amplitude decays as cos(t)
        """
        if not isinstance(amplitude, (float, int)) or amplitude <= 0:
            raise ForcingError(f"amplitude must be positive number, got {amplitude}")
        self.amplitude = float(amplitude)
        if not isinstance(k, int) or k < 1:
            raise ForcingError(f"k must be positive integer, got {k}")
        self.k = k
        self.time_dependent = time_dependent
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute Taylor-Green forcing."""
        # Time modulation if requested
        t = float(getattr(self, '_time', 0.0))  # Track time internally
        if self.time_dependent:
            amp = self.amplitude * jnp.cos(t)
        else:
            amp = self.amplitude
        
        # Create forcing in physical space
        kx = self.k * 2 * np.pi / grid.L
        forcing_phys = amp * jnp.sin(kx * grid.x) * jnp.cos(kx * grid.y)
        
        # Transform to spectral space
        forcing_hat = fft2(forcing_phys)
        
        # Update internal time
        self._time = t + dt
        
        return forcing_hat


class KolmogorovForcing(DeterministicForcing):
    """Kolmogorov flow forcing pattern.
    
    Implements forcing of the form:
    F = A * sin(k*y)  (for direction='y')
    F = A * sin(k*x)  (for direction='x')
    
    This creates parallel shear layers.
    
    Attributes:
        amplitude: Forcing amplitude
        k: Wavenumber
        direction: Direction of forcing ('x' or 'y')
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        k: int = 4,
        direction: Literal['x', 'y'] = 'y'
    ):
        """Initialize Kolmogorov forcing.
        
        Args:
            amplitude: Forcing amplitude
            k: Wavenumber (number of forcing bands)
            direction: Direction of sinusoidal forcing
        """
        if not isinstance(amplitude, (float, int)) or amplitude <= 0:
            raise ForcingError(f"amplitude must be positive number, got {amplitude}")
        self.amplitude = float(amplitude)
        if not isinstance(k, int) or k < 1:
            raise ForcingError(f"k must be positive integer, got {k}")
        self.k = k
        if direction not in ['x', 'y']:
            raise ForcingError(f"direction must be 'x' or 'y', got {direction}")
        self.direction = direction
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute Kolmogorov forcing."""
        kval = self.k * 2 * np.pi / grid.L
        
        if self.direction == 'y':
            forcing_phys = self.amplitude * jnp.sin(kval * grid.y)
        else:
            forcing_phys = self.amplitude * jnp.sin(kval * grid.x)
        
        return fft2(forcing_phys)


class CheckerboardForcing(DeterministicForcing):
    """Checkerboard pattern forcing.
    
    Creates alternating positive/negative forcing regions:
    F = A * sign(sin(kx*x) * sin(ky*y))
    
    Attributes:
        amplitude: Forcing amplitude
        kx: Wavenumber in x-direction
        ky: Wavenumber in y-direction
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        kx: int = 4,
        ky: Optional[int] = None
    ):
        """Initialize checkerboard forcing.
        
        Args:
            amplitude: Forcing amplitude
            kx: Number of cells in x-direction
            ky: Number of cells in y-direction (defaults to kx)
        """
        if not isinstance(amplitude, (float, int)) or amplitude <= 0:
            raise ForcingError(f"amplitude must be positive number, got {amplitude}")
        self.amplitude = float(amplitude)
        if not isinstance(kx, int) or kx < 1:
            raise ForcingError(f"kx must be positive integer, got {kx}")
        self.kx = kx
        self.ky = ky if ky is not None else kx
        if not isinstance(self.ky, int) or self.ky < 1:
            raise ForcingError(f"ky must be positive integer, got {self.ky}")
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute checkerboard forcing."""
        kx_val = self.kx * 2 * np.pi / grid.L
        ky_val = self.ky * 2 * np.pi / grid.L
        
        # Create checkerboard pattern
        pattern = jnp.sin(kx_val * grid.x) * jnp.sin(ky_val * grid.y)
        forcing_phys = self.amplitude * jnp.sign(pattern)
        
        return fft2(forcing_phys)


class ShearLayerForcing(DeterministicForcing):
    """Shear layer forcing pattern.
    
    Creates forcing that drives shear layers:
    - Linear shear: F = A * y/L (or x/L)
    - Hyperbolic tangent: F = A * tanh((y-y0)/δ)
    
    Attributes:
        amplitude: Forcing amplitude
        profile: Type of shear profile
        direction: Direction of shear
        center: Center position for tanh profile
        width: Width parameter for tanh profile
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        profile: Literal['linear', 'tanh'] = 'linear',
        direction: Literal['x', 'y'] = 'y',
        center: Optional[float] = None,
        width: float = 0.1
    ):
        """Initialize shear layer forcing.
        
        Args:
            amplitude: Forcing amplitude
            profile: Type of shear profile
            direction: Direction varying across shear
            center: Center of tanh profile (defaults to L/2)
            width: Width of tanh transition region
        """
        if not isinstance(amplitude, (float, int)) or amplitude <= 0:
            raise ForcingError(f"amplitude must be positive number, got {amplitude}")
        self.amplitude = float(amplitude)
        if profile not in ['linear', 'tanh']:
            raise ForcingError(f"profile must be 'linear' or 'tanh', got {profile}")
        self.profile = profile
        if direction not in ['x', 'y']:
            raise ForcingError(f"direction must be 'x' or 'y', got {direction}")
        self.direction = direction
        self.center = center
        if not isinstance(width, (float, int)) or width <= 0:
            raise ForcingError(f"width must be positive number, got {width}")
        self.width = float(width)
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute shear layer forcing."""
        if self.profile == 'linear':
            if self.direction == 'y':
                forcing_phys = self.amplitude * (grid.y / grid.L - 0.5)
            else:
                forcing_phys = self.amplitude * (grid.x / grid.L - 0.5)
        else:  # tanh profile
            center = self.center if self.center is not None else grid.L / 2
            if self.direction == 'y':
                forcing_phys = self.amplitude * jnp.tanh((grid.y - center) / self.width)
            else:
                forcing_phys = self.amplitude * jnp.tanh((grid.x - center) / self.width)
        
        return fft2(forcing_phys)


class VortexPairForcing(DeterministicForcing):
    """Vortex pair forcing pattern.
    
    Creates forcing from one or more point vortices or Gaussian vortices.
    
    Attributes:
        vortices: List of (x, y, circulation, radius) tuples
        amplitude_scale: Overall amplitude scaling
    """
    
    def __init__(
        self,
        vortices: list,
        amplitude_scale: float = 1.0
    ):
        """Initialize vortex pair forcing.
        
        Args:
            vortices: List of vortex specs (x, y, circulation, radius)
            amplitude_scale: Overall scaling factor
        """
        self.vortices = []
        for v in vortices:
            if len(v) != 4:
                raise ForcingError("Each vortex must be (x, y, circulation, radius)")
            x, y, circ, radius = v
            if not isinstance(radius, (float, int)) or radius <= 0:
                raise ForcingError(f"vortex radius must be positive, got {radius}")
            self.vortices.append({
                'x': float(x),
                'y': float(y),
                'circulation': float(circ),
                'radius': float(radius)
            })
        if not isinstance(amplitude_scale, (float, int)) or amplitude_scale <= 0:
            raise ForcingError(f"amplitude_scale must be positive number, got {amplitude_scale}")
        self.amplitude_scale = float(amplitude_scale)
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute vortex forcing."""
        forcing_phys = jnp.zeros_like(grid.x)
        
        for vortex in self.vortices:
            # Distance from vortex center (with periodic BC)
            dx = self._periodic_distance(grid.x, vortex['x'], grid.L)
            dy = self._periodic_distance(grid.y, vortex['y'], grid.L)
            r2 = dx**2 + dy**2
            
            # Gaussian vortex profile
            vorticity = (vortex['circulation'] / (np.pi * vortex['radius']**2) *
                        jnp.exp(-r2 / vortex['radius']**2))
            
            forcing_phys += vorticity
        
        forcing_phys *= self.amplitude_scale
        
        return fft2(forcing_phys)
    
    @staticmethod
    def _periodic_distance(coord: jax.Array, center: float, L: float) -> jax.Array:
        """Compute distance with periodic boundaries."""
        dist = coord - center
        # Handle periodic wraparound
        dist = jnp.where(dist > L/2, dist - L, dist)
        dist = jnp.where(dist < -L/2, dist + L, dist)
        return dist


class TimeModulatedForcing(DeterministicForcing):
    """Wrapper to add time modulation to any deterministic forcing.
    
    Allows modulating the amplitude of any base forcing pattern with
    various time-dependent functions.
    
    Attributes:
        base_forcing: Underlying deterministic forcing
        modulation: Type of time modulation
        frequency: Frequency for periodic modulations
        decay_rate: Decay rate for exponential modulation
    """
    
    def __init__(
        self,
        base_forcing: DeterministicForcing,
        modulation: Literal['sine', 'cosine', 'exponential', 'ramp'] = 'sine',
        frequency: float = 1.0,
        decay_rate: float = 1.0,
        phase: float = 0.0
    ):
        """Initialize time-modulated forcing.
        
        Args:
            base_forcing: Base deterministic forcing pattern
            modulation: Type of time modulation
            frequency: Frequency for sine/cosine modulation
            decay_rate: Rate for exponential decay
            phase: Phase shift for periodic modulations
        """
        if not isinstance(base_forcing, DeterministicForcing):
            raise ForcingError("base_forcing must be a DeterministicForcing instance")
        self.base_forcing = base_forcing
        if modulation not in ['sine', 'cosine', 'exponential', 'ramp']:
            raise ForcingError(f"Unknown modulation type: {modulation}")
        self.modulation = modulation
        if not isinstance(frequency, (float, int)) or frequency <= 0:
            raise ForcingError(f"frequency must be positive number, got {frequency}")
        self.frequency = float(frequency)
        if not isinstance(decay_rate, (float, int)) or decay_rate <= 0:
            raise ForcingError(f"decay_rate must be positive number, got {decay_rate}")
        self.decay_rate = float(decay_rate)
        self.phase = phase
        self._time = 0.0
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute time-modulated forcing."""
        # Get base forcing
        base_force = self.base_forcing(theta_hat, key, dt, grid)
        
        # Apply time modulation
        t = self._time
        if self.modulation == 'sine':
            factor = jnp.sin(2 * np.pi * self.frequency * t + self.phase)
        elif self.modulation == 'cosine':
            factor = jnp.cos(2 * np.pi * self.frequency * t + self.phase)
        elif self.modulation == 'exponential':
            factor = jnp.exp(-self.decay_rate * t)
        else:  # ramp
            factor = jnp.minimum(t * self.decay_rate, 1.0)
        
        # Update time
        self._time = t + dt
        
        return factor * base_force
    
    def get_diagnostics(self, theta_hat: jax.Array, forcing: jax.Array, grid: Grid) -> dict:
        """Get diagnostics including modulation info."""
        diags = self.base_forcing.get_diagnostics(theta_hat, forcing, grid)
        diags['modulation_type'] = self.modulation
        diags['current_time'] = self._time
        return diags


class CombinedDeterministicForcing(DeterministicForcing):
    """Combine multiple deterministic forcing patterns.
    
    Allows linear combination of different forcing patterns.
    
    Attributes:
        forcings: List of (weight, forcing) tuples
    """
    
    def __init__(self, forcings: list):
        """Initialize combined forcing.
        
        Args:
            forcings: List of (weight, DeterministicForcing) tuples
        """
        self.forcings = []
        for weight, forcing in forcings:
            if not isinstance(forcing, DeterministicForcing):
                raise ForcingError(
                    f"Each forcing must be DeterministicForcing, got {type(forcing)}"
                )
            self.forcings.append((float(weight), forcing))
    
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """Compute combined forcing."""
        total_forcing = jnp.zeros_like(theta_hat)
        
        for weight, forcing in self.forcings:
            total_forcing += weight * forcing(theta_hat, key, dt, grid)
        
        return total_forcing
    
    def get_diagnostics(self, theta_hat: jax.Array, forcing: jax.Array, grid: Grid) -> dict:
        """Get combined diagnostics."""
        diags = super().get_diagnostics(theta_hat, forcing, grid)
        diags['n_components'] = len(self.forcings)
        return diags


# Factory functions for common configurations

def make_taylor_green_forcing(
    amplitude: float = 1.0,
    k: int = 2,
    time_decay: bool = False
) -> DeterministicForcing:
    """Create Taylor-Green vortex forcing.
    
    Args:
        amplitude: Forcing amplitude
        k: Number of vortex pairs
        time_decay: Whether to apply cosine time decay
        
    Returns:
        Configured TaylorGreenForcing instance
    """
    return TaylorGreenForcing(amplitude=amplitude, k=k, time_dependent=time_decay)


def make_kolmogorov_forcing(
    amplitude: float = 1.0,
    k: int = 4,
    direction: Literal['x', 'y'] = 'y'
) -> DeterministicForcing:
    """Create Kolmogorov flow forcing.
    
    Args:
        amplitude: Forcing amplitude
        k: Number of forcing bands
        direction: Direction of sinusoidal forcing
        
    Returns:
        Configured KolmogorovForcing instance
    """
    return KolmogorovForcing(amplitude=amplitude, k=k, direction=direction)


def make_oscillating_forcing(
    base_pattern: str = 'taylor_green',
    frequency: float = 1.0,
    **kwargs
) -> DeterministicForcing:
    """Create time-oscillating deterministic forcing.
    
    Args:
        base_pattern: Type of base pattern
        frequency: Oscillation frequency
        **kwargs: Arguments passed to base pattern
        
    Returns:
        Time-modulated forcing instance
    """
    # Create base pattern
    if base_pattern == 'taylor_green':
        base = TaylorGreenForcing(**kwargs)
    elif base_pattern == 'kolmogorov':
        base = KolmogorovForcing(**kwargs)
    elif base_pattern == 'checkerboard':
        base = CheckerboardForcing(**kwargs)
    else:
        raise ForcingError(f"Unknown base pattern: {base_pattern}")
    
    # Wrap with sine modulation
    return TimeModulatedForcing(
        base_forcing=base,
        modulation='sine',
        frequency=frequency
    )
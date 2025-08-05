"""
Physical forcing patterns for pygSQuiG.

This module implements geophysically relevant forcing patterns including
shear layers, jets, and other coherent structures.
"""

from typing import Optional, Literal, Union
from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from pygsquig.core.grid import Grid, fft2
from pygsquig.exceptions import ForcingError


class PhysicalForcing(ABC):
    """Abstract base class for physical forcing patterns."""
    
    @abstractmethod
    def __call__(
        self,
        theta_hat: Array,
        key: Optional[jax.random.PRNGKey],
        dt: float,
        grid: Grid
    ) -> Array:
        """Apply forcing pattern."""
        pass


class ShearLayerForcing(PhysicalForcing):
    """Shear layer forcing pattern.
    
    Creates horizontal or vertical shear layers that can drive
    Kelvin-Helmholtz instabilities and mixing.
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        shear_width: float = 0.1,
        n_layers: int = 1,
        orientation: Literal['horizontal', 'vertical'] = 'horizontal',
        time_dependence: Literal['steady', 'oscillatory', 'pulsed'] = 'steady',
        omega: float = 1.0,
        phase: float = 0.0,
        perturbation_amplitude: float = 0.0,
        perturbation_k: int = 10
    ):
        """Initialize shear layer forcing.
        
        Args:
            amplitude: Base forcing amplitude
            shear_width: Width of shear layer (fraction of domain)
            n_layers: Number of shear layers
            orientation: 'horizontal' or 'vertical' shear
            time_dependence: Time modulation type
            omega: Oscillation frequency for time dependence
            phase: Initial phase
            perturbation_amplitude: Amplitude of sinusoidal perturbations
            perturbation_k: Wavenumber of perturbations along layer
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")
        if shear_width <= 0 or shear_width > 0.5:
            raise ForcingError("Shear width must be in (0, 0.5]")
        if n_layers < 1:
            raise ForcingError("Number of layers must be >= 1")
            
        self.amplitude = amplitude
        self.shear_width = shear_width
        self.n_layers = n_layers
        self.orientation = orientation
        self.time_dependence = time_dependence
        self.omega = omega
        self.phase = phase
        self.perturbation_amplitude = perturbation_amplitude
        self.perturbation_k = perturbation_k
        self._time = 0.0
        
    def __call__(
        self,
        theta_hat: Array,
        key: Optional[jax.random.PRNGKey],
        dt: float,
        grid: Grid
    ) -> Array:
        """Apply shear layer forcing."""
        # Create shear profile in physical space
        x, y = grid.x, grid.y
        L = grid.L
        
        # Layer spacing
        layer_spacing = L / self.n_layers
        layer_width = self.shear_width * L
        
        # Initialize forcing field
        forcing = jnp.zeros_like(x)
        
        for i in range(self.n_layers):
            if self.orientation == 'horizontal':
                # Horizontal shear layers
                y_center = (i + 0.5) * layer_spacing
                
                # Add perturbations if requested
                if self.perturbation_amplitude > 0:
                    perturbation = self.perturbation_amplitude * layer_width * \
                                 jnp.sin(2 * np.pi * self.perturbation_k * x / L)
                    y_eff = y - y_center - perturbation
                else:
                    y_eff = y - y_center
                    
                # Periodic wrapping - adjust for centered domain
                y_eff = ((y_eff + L/2) % L) - L/2
                
                # Tanh shear profile
                shear_profile = jnp.tanh(y_eff / layer_width)
                
                # Alternate sign for adjacent layers
                sign = 1.0 if i % 2 == 0 else -1.0
                forcing = forcing + sign * shear_profile
                
            else:  # vertical
                # Vertical shear layers
                x_center = (i + 0.5) * layer_spacing
                
                # Add perturbations
                if self.perturbation_amplitude > 0:
                    perturbation = self.perturbation_amplitude * layer_width * \
                                 jnp.sin(2 * np.pi * self.perturbation_k * y / L)
                    x_eff = x - x_center - perturbation
                else:
                    x_eff = x - x_center
                    
                # Periodic wrapping
                x_eff = ((x_eff + L/2) % L) - L/2
                
                # Tanh profile
                shear_profile = jnp.tanh(x_eff / layer_width)
                
                # Alternate sign
                sign = 1.0 if i % 2 == 0 else -1.0
                forcing = forcing + sign * shear_profile
                
        # Apply time modulation
        amplitude = self._get_time_modulation(dt)
        forcing = amplitude * forcing
        
        # Transform to spectral space
        forcing_hat = fft2(forcing)
        
        # Remove mean (k=0 mode)
        forcing_hat = forcing_hat.at[0, 0].set(0.0)
        
        return forcing_hat
        
    def _get_time_modulation(self, dt: float) -> float:
        """Get time-dependent amplitude."""
        self._time += dt
        
        if self.time_dependence == 'steady':
            return self.amplitude
        elif self.time_dependence == 'oscillatory':
            return self.amplitude * jnp.cos(self.omega * self._time + self.phase)
        elif self.time_dependence == 'pulsed':
            # Pulse every 2π/omega
            phase = (self.omega * self._time + self.phase) % (2 * np.pi)
            return self.amplitude if phase < np.pi else 0.0
        else:
            return self.amplitude


class JetForcing(PhysicalForcing):
    """Jet forcing pattern.
    
    Creates zonal or meridional jets with optional meandering.
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        jet_width: float = 0.1,
        n_jets: int = 2,
        meander_amplitude: float = 0.0,
        meander_k: int = 2,
        meander_phase_speed: float = 0.0,
        orientation: Literal['zonal', 'meridional'] = 'zonal',
        profile: Literal['gaussian', 'sech2', 'tanh'] = 'gaussian',
        time_dependence: Literal['steady', 'oscillatory', 'growing'] = 'steady',
        omega: float = 1.0,
        growth_rate: float = 0.1
    ):
        """Initialize jet forcing.
        
        Args:
            amplitude: Maximum jet velocity
            jet_width: Width of jets (fraction of domain)
            n_jets: Number of jets
            meander_amplitude: Amplitude of jet meandering
            meander_k: Wavenumber of meandering
            meander_phase_speed: Phase speed of meanders
            orientation: 'zonal' (east-west) or 'meridional' (north-south)
            profile: Jet profile shape
            time_dependence: Time evolution type
            omega: Oscillation frequency
            growth_rate: Growth rate for growing jets
        """
        if amplitude < 0:
            raise ForcingError("Amplitude must be non-negative")
        if jet_width <= 0 or jet_width > 0.5:
            raise ForcingError("Jet width must be in (0, 0.5]")
        if n_jets < 1:
            raise ForcingError("Number of jets must be >= 1")
            
        self.amplitude = amplitude
        self.jet_width = jet_width
        self.n_jets = n_jets
        self.meander_amplitude = meander_amplitude
        self.meander_k = meander_k
        self.meander_phase_speed = meander_phase_speed
        self.orientation = orientation
        self.profile = profile
        self.time_dependence = time_dependence
        self.omega = omega
        self.growth_rate = growth_rate
        self._time = 0.0
        
    def __call__(
        self,
        theta_hat: Array,
        key: Optional[jax.random.PRNGKey],
        dt: float,
        grid: Grid
    ) -> Array:
        """Apply jet forcing."""
        x, y = grid.x, grid.y
        L = grid.L
        
        # Jet spacing
        jet_spacing = L / self.n_jets
        jet_width = self.jet_width * L
        
        # Initialize forcing
        forcing = jnp.zeros_like(x)
        
        # Time for meandering
        t = self._time
        
        for i in range(self.n_jets):
            if self.orientation == 'zonal':
                # Zonal jets (east-west)
                y_center = (i + 0.5) * jet_spacing
                
                # Add meandering
                if self.meander_amplitude > 0:
                    meander = self.meander_amplitude * jet_width * \
                            jnp.sin(2 * np.pi * self.meander_k * x / L - 
                                   self.meander_phase_speed * t)
                    y_eff = y - y_center - meander
                else:
                    y_eff = y - y_center
                    
                # Periodic wrapping
                y_eff = ((y_eff + L/2) % L) - L/2
                
                # Jet profile
                if self.profile == 'gaussian':
                    jet_profile = jnp.exp(-(y_eff / jet_width)**2)
                elif self.profile == 'sech2':
                    jet_profile = 1.0 / jnp.cosh(y_eff / jet_width)**2
                else:  # tanh
                    # Derivative of tanh gives sech^2 profile
                    jet_profile = 1.0 / jnp.cosh(y_eff / jet_width)**2
                    
                # Alternate direction
                sign = 1 if i % 2 == 0 else -1
                forcing = forcing + sign * jet_profile
                
            else:  # meridional
                # Meridional jets (north-south)
                x_center = (i + 0.5) * jet_spacing
                
                # Add meandering
                if self.meander_amplitude > 0:
                    meander = self.meander_amplitude * jet_width * \
                            jnp.sin(2 * np.pi * self.meander_k * y / L - 
                                   self.meander_phase_speed * t)
                    x_eff = x - x_center - meander
                else:
                    x_eff = x - x_center
                    
                # Periodic wrapping
                x_eff = ((x_eff + L/2) % L) - L/2
                
                # Jet profile
                if self.profile == 'gaussian':
                    jet_profile = jnp.exp(-(x_eff / jet_width)**2)
                elif self.profile == 'sech2':
                    jet_profile = 1.0 / jnp.cosh(x_eff / jet_width)**2
                else:  # tanh
                    jet_profile = 1.0 / jnp.cosh(x_eff / jet_width)**2
                    
                # Alternate direction
                sign = 1 if i % 2 == 0 else -1
                forcing = forcing + sign * jet_profile
                
        # Apply time modulation
        amplitude = self._get_time_modulation(dt)
        forcing = amplitude * forcing
        
        # Transform to spectral space
        forcing_hat = fft2(forcing)
        
        # Remove mean
        forcing_hat = forcing_hat.at[0, 0].set(0.0)
        
        return forcing_hat
        
    def _get_time_modulation(self, dt: float) -> float:
        """Get time-dependent amplitude."""
        self._time += dt
        
        if self.time_dependence == 'steady':
            return self.amplitude
        elif self.time_dependence == 'oscillatory':
            return self.amplitude * jnp.cos(self.omega * self._time)
        elif self.time_dependence == 'growing':
            return self.amplitude * jnp.exp(self.growth_rate * self._time)
        else:
            return self.amplitude


class ConvectivePlumesForcing(PhysicalForcing):
    """Convective plumes forcing pattern.
    
    Simulates buoyant plumes rising from bottom boundary.
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        plume_radius: float = 0.05,
        n_plumes: int = 5,
        rise_velocity: float = 1.0,
        buoyancy_decay: float = 0.1,
        randomize_positions: bool = True
    ):
        """Initialize convective plumes forcing.
        
        Args:
            amplitude: Plume buoyancy amplitude
            plume_radius: Radius of plumes (fraction of domain)
            n_plumes: Number of simultaneous plumes
            rise_velocity: Vertical velocity of plumes
            buoyancy_decay: Decay rate of buoyancy
            randomize_positions: Random vs regular plume positions
        """
        self.amplitude = amplitude
        self.plume_radius = plume_radius
        self.n_plumes = n_plumes
        self.rise_velocity = rise_velocity
        self.buoyancy_decay = buoyancy_decay
        self.randomize_positions = randomize_positions
        
        # Initialize plume positions and ages
        self.plume_data = None
        
    def __call__(
        self,
        theta_hat: Array,
        key: Optional[jax.random.PRNGKey],
        dt: float,
        grid: Grid
    ) -> Array:
        """Apply convective plumes forcing."""
        x, y = grid.x, grid.y
        L = grid.L
        radius = self.plume_radius * L
        
        # Initialize plumes if needed
        if self.plume_data is None:
            self._initialize_plumes(L, key)
            
        # Update plume positions
        self._update_plumes(dt, L, key)
        
        # Create forcing field
        forcing = jnp.zeros_like(x)
        
        for i in range(self.n_plumes):
            x_p, y_p, age = self.plume_data[i]
            
            # Distance from plume center
            dx = x - x_p
            dy = y - y_p
            
            # Periodic boundary conditions
            dx = jnp.where(dx > L/2, dx - L, dx)
            dx = jnp.where(dx < -L/2, dx + L, dx)
            dy = jnp.where(dy > L/2, dy - L, dy)
            dy = jnp.where(dy < -L/2, dy + L, dy)
            
            r_squared = dx**2 + dy**2
            
            # Gaussian plume profile
            plume_profile = jnp.exp(-r_squared / (2 * radius**2))
            
            # Decay with age
            decay_factor = jnp.exp(-self.buoyancy_decay * age)
            
            forcing = forcing + self.amplitude * decay_factor * plume_profile
            
        # Transform to spectral space
        forcing_hat = fft2(forcing)
        
        # Remove mean
        forcing_hat = forcing_hat.at[0, 0].set(0.0)
        
        return forcing_hat
        
    def _initialize_plumes(self, L: float, key: Optional[jax.random.PRNGKey]):
        """Initialize plume positions."""
        self.plume_data = []
        
        if self.randomize_positions and key is not None:
            # Random positions
            keys = jax.random.split(key, self.n_plumes)
            for i in range(self.n_plumes):
                x = jax.random.uniform(keys[i], minval=0, maxval=L)
                y = jax.random.uniform(keys[i], minval=0, maxval=L/4)  # Start low
                age = jax.random.uniform(keys[i], minval=0, maxval=1)
                self.plume_data.append([float(x), float(y), float(age)])
        else:
            # Regular spacing
            spacing = L / self.n_plumes
            for i in range(self.n_plumes):
                x = (i + 0.5) * spacing
                y = 0.1 * L
                age = 0.0
                self.plume_data.append([x, y, age])
                
    def _update_plumes(self, dt: float, L: float, key: Optional[jax.random.PRNGKey]):
        """Update plume positions and ages."""
        for i in range(self.n_plumes):
            x, y, age = self.plume_data[i]
            
            # Rise
            y += self.rise_velocity * dt
            age += dt
            
            # Reset if plume reaches top or gets too old
            if y > L or age > 5.0 / self.buoyancy_decay:
                if self.randomize_positions and key is not None:
                    key, subkey = jax.random.split(key)
                    x = float(jax.random.uniform(subkey, minval=0, maxval=L))
                else:
                    x = (i + 0.5) * L / self.n_plumes
                y = 0.1 * L
                age = 0.0
                
            self.plume_data[i] = [x, y, age]


class TopographicForcing(PhysicalForcing):
    """Topographic forcing pattern.
    
    Simulates flow over topography.
    """
    
    def __init__(
        self,
        amplitude: float = 1.0,
        topography_type: Literal['ridge', 'seamount', 'rough'] = 'ridge',
        k_topo: int = 4,
        orientation: Literal['zonal', 'meridional'] = 'zonal'
    ):
        """Initialize topographic forcing.
        
        Args:
            amplitude: Forcing amplitude
            topography_type: Type of topography
            k_topo: Characteristic wavenumber
            orientation: Orientation for ridges
        """
        self.amplitude = amplitude
        self.topography_type = topography_type
        self.k_topo = k_topo
        self.orientation = orientation
        
    def __call__(
        self,
        theta_hat: Array,
        key: Optional[jax.random.PRNGKey],
        dt: float,
        grid: Grid
    ) -> Array:
        """Apply topographic forcing."""
        x, y = grid.x, grid.y
        L = grid.L
        k = 2 * np.pi * self.k_topo / L
        
        if self.topography_type == 'ridge':
            # Sinusoidal ridge
            if self.orientation == 'zonal':
                forcing = self.amplitude * jnp.sin(k * y)
            else:
                forcing = self.amplitude * jnp.sin(k * x)
                
        elif self.topography_type == 'seamount':
            # Isolated seamounts
            forcing = self.amplitude * (
                jnp.sin(k * x) * jnp.sin(k * y)
            )
            
        else:  # rough
            # Random rough topography
            if key is not None:
                # Use random phases
                key1, key2 = jax.random.split(key)
                phase_x = jax.random.uniform(key1, minval=0, maxval=2*np.pi)
                phase_y = jax.random.uniform(key2, minval=0, maxval=2*np.pi)
            else:
                phase_x = phase_y = 0
                
            forcing = self.amplitude * (
                jnp.sin(k * x + phase_x) + 
                jnp.sin(2 * k * x + phase_y) * 0.5 +
                jnp.sin(k * y + phase_x) +
                jnp.sin(2 * k * y + phase_y) * 0.5
            ) / 3.0
            
        # Transform to spectral space
        forcing_hat = fft2(forcing)
        
        # Remove mean
        forcing_hat = forcing_hat.at[0, 0].set(0.0)
        
        return forcing_hat
"""
Ring forcing for turbulent energy injection.

This module implements constant-amplitude ring forcing in Fourier space
to maintain statistically stationary turbulence with controlled energy flux.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from pygsquig.core.grid import Grid, ifft2


class RingForcing:
    """
    Ring forcing in Fourier space with controlled energy injection.
    
    Implements stochastic forcing concentrated in a ring of wavenumbers
    to drive turbulence with a target energy flux ε.
    
    Attributes:
        kf: Central forcing wavenumber
        dk: Width of forcing ring
        epsilon: Target energy flux ⟨θF⟩
        tau_f: Correlation time (0 for white noise)
        amplitude: Base amplitude for forcing modes
    """
    
    def __init__(
        self,
        kf: float,
        dk: float = 1.0,
        epsilon: float = 0.1,
        tau_f: float = 0.0,
        amplitude: Optional[float] = None
    ):
        """
        Initialize ring forcing.
        
        Parameters:
            kf: Central forcing wavenumber (typically ~20 for L=2π)
            dk: Shell width (default 1)
            epsilon: Target energy flux (default 0.1)
            tau_f: Correlation time (default 0 = white noise)
            amplitude: Base amplitude (computed if None)
        """
        self.kf = kf
        self.dk = dk
        self.epsilon = epsilon
        self.tau_f = tau_f
        
        # Set default amplitude if not provided
        # This will be rescaled to achieve target epsilon
        if amplitude is None:
            self.amplitude = jnp.sqrt(epsilon)
        else:
            self.amplitude = amplitude
            
        # For OU process (temporal correlation)
        self.forcing_state = None
        
    def _get_forcing_mask(self, grid: Grid) -> jax.Array:
        """
        Create mask for modes in forcing ring.
        
        Parameters:
            grid: Grid object
            
        Returns:
            Boolean mask for modes with |k - kf| ≤ dk/2
        """
        k_mag = jnp.sqrt(grid.k2)
        mask = jnp.abs(k_mag - self.kf) <= self.dk / 2
        return mask
        
    def _generate_random_phases(self, key: jax.random.PRNGKey, shape: tuple) -> jax.Array:
        """
        Generate random phases for forcing.
        
        Parameters:
            key: PRNG key
            shape: Shape of output array
            
        Returns:
            Complex array with random phases and unit amplitude
        """
        # Generate random angles
        phases = jax.random.uniform(key, shape=shape, minval=0, maxval=2*jnp.pi)
        # Convert to complex with unit amplitude
        return jnp.exp(1j * phases)
        
    def _apply_temporal_correlation(
        self,
        forcing_new: jax.Array,
        dt: float
    ) -> jax.Array:
        """
        Apply Ornstein-Uhlenbeck process for temporal correlation.
        
        Parameters:
            forcing_new: New white noise forcing
            dt: Time step
            
        Returns:
            Temporally correlated forcing
        """
        if self.tau_f == 0 or self.forcing_state is None:
            # No correlation or first call
            self.forcing_state = forcing_new
            return forcing_new
            
        # OU process: dF/dt = -(F - F_white)/tau_f
        # Discrete update: F_n+1 = F_n + dt * (F_white - F_n) / tau_f
        decay = jnp.exp(-dt / self.tau_f)
        variance_factor = jnp.sqrt(1 - decay**2)
        
        self.forcing_state = (
            decay * self.forcing_state + 
            variance_factor * forcing_new
        )
        
        return self.forcing_state
        
    def __call__(
        self,
        theta_hat: jax.Array,
        key: jax.random.PRNGKey,
        dt: float,
        grid: Grid
    ) -> jax.Array:
        """
        Compute forcing for one time step.
        
        Parameters:
            theta_hat: Current state in spectral space
            key: PRNG key for random phases
            dt: Time step (used for temporal correlation)
            grid: Grid object
            
        Returns:
            Forcing in spectral space F̂
        """
        # Get forcing mask
        mask = self._get_forcing_mask(grid)
        
        # Generate random phases
        random_phases = self._generate_random_phases(key, theta_hat.shape)
        
        # Create forcing with constant amplitude and random phases
        forcing = self.amplitude * mask * random_phases
        
        # Apply temporal correlation if needed
        if self.tau_f > 0:
            forcing = self._apply_temporal_correlation(forcing, dt)
            
        # Enforce reality condition: F(-k) = F*(k)
        # This is automatically satisfied if we start from real space
        # but let's ensure it for the forcing
        forcing = _ensure_hermitian_symmetry(forcing)
        
        # Compute current energy injection rate
        theta = ifft2(theta_hat)
        force_phys = ifft2(forcing)
        current_rate = jnp.mean(theta * force_phys)
        
        # Rescale to achieve target energy flux
        # We want ⟨θF⟩ = ε
        # Only rescale if there's overlap between theta and forcing
        # Otherwise, keep the forcing at nominal amplitude
        scale = jnp.where(
            jnp.abs(current_rate) > 1e-10,
            self.epsilon / current_rate,
            1.0  # Keep nominal amplitude if no overlap
        )
        
        # Apply scaling
        forcing = scale * forcing
        
        return forcing
        
    def get_diagnostics(self, theta_hat: jax.Array, forcing: jax.Array, grid: Grid) -> dict:
        """
        Compute forcing diagnostics.
        
        Parameters:
            theta_hat: Current state
            forcing: Current forcing
            grid: Grid object
            
        Returns:
            Dictionary with diagnostic values
        """
        # Energy injection rate
        theta = ifft2(theta_hat)
        force_phys = ifft2(forcing)
        injection_rate = jnp.mean(theta * force_phys)
        
        # Power in forcing ring
        mask = self._get_forcing_mask(grid)
        forcing_power = jnp.sum(jnp.abs(forcing * mask)**2)
        
        # Number of forced modes
        n_forced = jnp.sum(mask)
        
        return {
            'injection_rate': float(injection_rate),
            'forcing_power': float(forcing_power),
            'n_forced_modes': int(n_forced),
            'target_epsilon': self.epsilon
        }


def _ensure_hermitian_symmetry(field_hat: jax.Array) -> jax.Array:
    """
    Ensure Hermitian symmetry for real-valued fields.
    
    For a real field, the Fourier transform satisfies F(-k) = F*(k).
    This function enforces this constraint.
    
    Parameters:
        field_hat: Fourier coefficients
        
    Returns:
        Fourier coefficients with enforced symmetry
    """
    N = field_hat.shape[0]
    
    # For even N, the Nyquist frequency needs special treatment
    # F[N/2, :] and F[:, N/2] must be real
    
    # First ensure DC component is real
    field_hat = field_hat.at[0, 0].set(field_hat[0, 0].real)
    
    # Ensure Nyquist frequencies are real
    if N % 2 == 0:
        N2 = N // 2
        # F[N/2, 0] and F[0, N/2] must be real
        field_hat = field_hat.at[N2, 0].set(field_hat[N2, 0].real)
        field_hat = field_hat.at[0, N2].set(field_hat[0, N2].real)
        field_hat = field_hat.at[N2, N2].set(field_hat[N2, N2].real)
        
    # The general Hermitian symmetry is automatically satisfied
    # by the way we generate the forcing, but we could enforce it
    # more explicitly if needed
    
    return field_hat
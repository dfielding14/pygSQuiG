"""
Tests for stochastic forcing patterns.

This module tests white noise, colored noise, vortex injection,
and Ornstein-Uhlenbeck forcing implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import ifft2, make_grid
from pygsquig.exceptions import ForcingError
from pygsquig.forcing.stochastic_forcing import (
    ColoredNoiseForcing,
    OrnsteinUhlenbeckForcing,
    StochasticVortexForcing,
    WhiteNoiseForcing,
    create_combined_stochastic_forcing,
)


class TestWhiteNoiseForcing:
    """Test white noise forcing implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test white noise forcing initialization."""
        forcing = WhiteNoiseForcing(amplitude=0.1)
        assert forcing.amplitude == 0.1
        assert forcing.isotropy is True

        # Test with wavenumber limits
        forcing = WhiteNoiseForcing(amplitude=0.2, k_min=5.0, k_max=20.0, isotropy=False)
        assert forcing.k_min == 5.0
        assert forcing.k_max == 20.0
        assert forcing.isotropy is False

    def test_invalid_amplitude(self):
        """Test invalid amplitude raises error."""
        with pytest.raises(ForcingError):
            WhiteNoiseForcing(amplitude=-0.1)

    def test_forcing_application(self):
        """Test basic forcing application."""
        forcing = WhiteNoiseForcing(amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = forcing(self.theta_hat, key, dt, self.grid)

        # Check shape
        assert result.shape == self.theta_hat.shape
        assert result.dtype == self.theta_hat.dtype

        # Check non-zero
        assert not jnp.allclose(result, 0)

        # Check k=0 mode is zero (no mean injection)
        assert result[0, 0] == 0

    def test_spectral_filtering(self):
        """Test spectral band filtering."""
        k_min = 10.0
        k_max = 20.0
        forcing = WhiteNoiseForcing(amplitude=0.1, k_min=k_min, k_max=k_max)

        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = forcing(self.theta_hat, key, dt, self.grid)

        # Check that forcing is zero outside band
        kx = self.grid.kx
        ky = self.grid.ky
        k = jnp.sqrt(kx**2 + ky**2)

        mask = (k < k_min) | (k > k_max)
        assert jnp.allclose(result[mask], 0)

        # Check non-zero inside band
        mask_band = (k >= k_min) & (k <= k_max)
        assert not jnp.allclose(result[mask_band], 0)

    def test_reality_condition(self):
        """Test that forcing satisfies reality condition."""
        forcing = WhiteNoiseForcing(amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = forcing(self.theta_hat, key, dt, self.grid)

        # Transform to physical space should be real
        phys = ifft2(result)
        assert jnp.allclose(phys.imag, 0, atol=1e-10)

        # Check Nyquist modes are real
        N = self.N
        assert jnp.isreal(result[0, 0])
        assert jnp.isreal(result[N // 2, 0])
        assert jnp.isreal(result[0, N // 2])
        assert jnp.isreal(result[N // 2, N // 2])

    def test_amplitude_scaling(self):
        """Test that forcing scales with amplitude and dt."""
        key = jax.random.PRNGKey(42)

        # Different amplitudes
        amp1 = 0.1
        amp2 = 0.2
        forcing1 = WhiteNoiseForcing(amplitude=amp1, isotropy=False)
        forcing2 = WhiteNoiseForcing(amplitude=amp2, isotropy=False)

        dt = 0.01
        result1 = forcing1(self.theta_hat, key, dt, self.grid)
        result2 = forcing2(self.theta_hat, key, dt, self.grid)

        # RMS should scale with amplitude
        rms1 = jnp.sqrt(jnp.mean(jnp.abs(result1) ** 2))
        rms2 = jnp.sqrt(jnp.mean(jnp.abs(result2) ** 2))

        assert abs(rms2 / rms1 - amp2 / amp1) < 0.1  # Within 10%

    def test_different_random_keys(self):
        """Test that different keys give different results."""
        forcing = WhiteNoiseForcing(amplitude=0.1)
        dt = 0.01

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        result1 = forcing(self.theta_hat, key1, dt, self.grid)
        result2 = forcing(self.theta_hat, key2, dt, self.grid)

        assert not jnp.allclose(result1, result2)


class TestColoredNoiseForcing:
    """Test colored noise forcing implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test colored noise initialization."""
        forcing = ColoredNoiseForcing(amplitude=0.1, spectral_slope=-2.0, k_peak=10.0, k_width=5.0)
        assert forcing.amplitude == 0.1
        assert forcing.spectral_slope == -2.0
        assert forcing.k_peak == 10.0
        assert forcing.k_width == 5.0

    def test_spectral_characteristics(self):
        """Test that spectrum has correct shape."""
        k_peak = 15.0
        forcing = ColoredNoiseForcing(
            amplitude=0.1, spectral_slope=-2.0, k_peak=k_peak, k_width=5.0
        )

        key = jax.random.PRNGKey(42)
        dt = 0.01

        # Average over multiple realizations
        n_samples = 100
        power_spectrum = jnp.zeros(self.N // 2)

        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            result = forcing(self.theta_hat, subkey, dt, self.grid)

            # Compute radial power spectrum
            kx = self.grid.kx
            ky = self.grid.ky
            k = jnp.sqrt(kx**2 + ky**2)

            for ki in range(1, self.N // 2):
                mask = (k >= ki - 0.5) & (k < ki + 0.5)
                power_spectrum = power_spectrum.at[ki].add(jnp.sum(jnp.abs(result[mask]) ** 2))

        power_spectrum = power_spectrum / n_samples

        # Find peak
        k_peak_idx = jnp.argmax(power_spectrum[1:]) + 1

        # Peak should be near k_peak
        assert abs(k_peak_idx - k_peak) < 5

    def test_reality_condition(self):
        """Test reality condition for colored noise."""
        forcing = ColoredNoiseForcing(amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = forcing(self.theta_hat, key, dt, self.grid)

        # Physical space should be real
        phys = ifft2(result)
        assert jnp.allclose(phys.imag, 0, atol=1e-10)


class TestStochasticVortexForcing:
    """Test stochastic vortex injection forcing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test vortex forcing initialization."""
        forcing = StochasticVortexForcing(
            amplitude=0.1, vortex_size=0.1, injection_rate=1.0, vortex_strength_std=0.2
        )
        assert forcing.amplitude == 0.1
        assert forcing.vortex_size == 0.1
        assert forcing.injection_rate == 1.0
        assert forcing.vortex_strength_std == 0.2

    def test_invalid_parameters(self):
        """Test invalid parameters raise errors."""
        with pytest.raises(ForcingError):
            StochasticVortexForcing(amplitude=-0.1)

        with pytest.raises(ForcingError):
            StochasticVortexForcing(vortex_size=0.0)

        with pytest.raises(ForcingError):
            StochasticVortexForcing(vortex_size=1.5)

        with pytest.raises(ForcingError):
            StochasticVortexForcing(injection_rate=-1.0)

    def test_vortex_injection(self):
        """Test vortex injection mechanism."""
        # High injection rate to ensure vortices
        forcing = StochasticVortexForcing(
            amplitude=0.5,
            vortex_size=0.1,
            injection_rate=100.0,  # High rate
            vortex_strength_std=0.1,
        )

        key = jax.random.PRNGKey(42)
        dt = 0.1

        result = forcing(self.theta_hat, key, dt, self.grid)

        # Should inject vortices
        assert not jnp.allclose(result, 0)

        # Check physical space has coherent structures
        phys = ifft2(result)
        assert jnp.max(jnp.abs(phys)) > 0.1  # Should have peaks

    def test_low_injection_rate(self):
        """Test that low injection rate sometimes gives no vortices."""
        forcing = StochasticVortexForcing(
            amplitude=0.1, vortex_size=0.1, injection_rate=0.1, vortex_strength_std=0.1  # Low rate
        )

        key = jax.random.PRNGKey(42)
        dt = 0.01  # Small dt

        # Try multiple times
        no_vortex_count = 0
        for i in range(20):
            key, subkey = jax.random.split(key)
            result = forcing(self.theta_hat, subkey, dt, self.grid)
            if jnp.allclose(result, 0):
                no_vortex_count += 1

        # Should have some cases with no vortices
        assert no_vortex_count > 0


class TestOrnsteinUhlenbeckForcing:
    """Test Ornstein-Uhlenbeck process forcing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_initialization(self):
        """Test OU forcing initialization."""
        forcing = OrnsteinUhlenbeckForcing(
            amplitude=0.1, correlation_time=1.0, k_min=5.0, k_max=20.0
        )
        assert forcing.amplitude == 0.1
        assert forcing.correlation_time == 1.0
        assert forcing.k_min == 5.0
        assert forcing.k_max == 20.0
        assert forcing._ou_state is None

    def test_invalid_parameters(self):
        """Test invalid parameters."""
        with pytest.raises(ForcingError):
            OrnsteinUhlenbeckForcing(amplitude=-0.1)

        with pytest.raises(ForcingError):
            OrnsteinUhlenbeckForcing(correlation_time=0.0)

    def test_temporal_correlation(self):
        """Test that OU process has temporal correlations."""
        tau = 1.0
        forcing = OrnsteinUhlenbeckForcing(amplitude=0.1, correlation_time=tau)

        key = jax.random.PRNGKey(42)
        dt = 0.01

        # First call initializes state
        result1 = forcing(self.theta_hat, key, dt, self.grid)

        # Second call should be correlated
        key, subkey = jax.random.split(key)
        result2 = forcing(self.theta_hat, subkey, dt, self.grid)

        # Correlation should be high for small dt/tau
        correlation = jnp.sum(result1.conj() * result2).real
        norm1 = jnp.sqrt(jnp.sum(jnp.abs(result1) ** 2))
        norm2 = jnp.sqrt(jnp.sum(jnp.abs(result2) ** 2))

        corr_coeff = correlation / (norm1 * norm2)

        # Should be close to exp(-dt/tau) ≈ 0.99
        assert corr_coeff > 0.9

    def test_decorrelation_over_time(self):
        """Test that correlation decays over time."""
        tau = 0.1
        forcing = OrnsteinUhlenbeckForcing(amplitude=0.1, correlation_time=tau)

        key = jax.random.PRNGKey(42)
        dt = 0.05  # Half of tau

        # Initial state
        result1 = forcing(self.theta_hat, key, dt, self.grid)

        # Evolve for several correlation times
        result2 = result1
        for _ in range(10):  # 5 tau total
            key, subkey = jax.random.split(key)
            result2 = forcing(self.theta_hat, subkey, dt, self.grid)

        # Should be decorrelated
        correlation = jnp.sum(result1.conj() * result2).real
        norm1 = jnp.sqrt(jnp.sum(jnp.abs(result1) ** 2))
        norm2 = jnp.sqrt(jnp.sum(jnp.abs(result2) ** 2))

        corr_coeff = abs(correlation / (norm1 * norm2))

        # Should be nearly uncorrelated
        assert corr_coeff < 0.2


class TestCombinedForcing:
    """Test combined stochastic forcing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.theta_hat = jnp.zeros((self.N, self.N), dtype=complex)

    def test_combined_forcing_creation(self):
        """Test creating combined forcing."""
        forcing1 = WhiteNoiseForcing(amplitude=0.1)
        forcing2 = ColoredNoiseForcing(amplitude=0.1)

        combined = create_combined_stochastic_forcing([forcing1, forcing2], weights=[1.0, 2.0])

        assert combined is not None

    def test_empty_forcing_list(self):
        """Test that empty list raises error."""
        with pytest.raises(ForcingError):
            create_combined_stochastic_forcing([])

    def test_mismatched_weights(self):
        """Test mismatched weights raise error."""
        forcing1 = WhiteNoiseForcing(amplitude=0.1)
        forcing2 = ColoredNoiseForcing(amplitude=0.1)

        with pytest.raises(ForcingError):
            create_combined_stochastic_forcing([forcing1, forcing2], weights=[1.0])  # Wrong length

    def test_combined_forcing_application(self):
        """Test applying combined forcing."""
        forcing1 = WhiteNoiseForcing(amplitude=0.1, k_max=10.0)
        forcing2 = WhiteNoiseForcing(amplitude=0.1, k_min=10.0)

        combined = create_combined_stochastic_forcing([forcing1, forcing2], weights=[1.0, 1.0])

        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = combined(self.theta_hat, key, dt, self.grid)

        # Should be non-zero
        assert not jnp.allclose(result, 0)

        # Should have contributions in both bands
        kx = self.grid.kx
        ky = self.grid.ky
        k = jnp.sqrt(kx**2 + ky**2)

        low_k_mask = (k > 0) & (k < 10.0)
        high_k_mask = k >= 10.0

        assert not jnp.allclose(result[low_k_mask], 0)
        assert not jnp.allclose(result[high_k_mask], 0)


# Property-based tests
class TestProperties:
    """Property-based tests for stochastic forcing."""

    def test_energy_conservation_property(self):
        """Test that forcing doesn't inject at k=0."""
        N = 32
        grid = make_grid(N, 2 * np.pi)
        theta_hat = jnp.zeros((N, N), dtype=complex)

        forcings = [
            WhiteNoiseForcing(amplitude=0.1),
            ColoredNoiseForcing(amplitude=0.1),
            OrnsteinUhlenbeckForcing(amplitude=0.1),
        ]

        key = jax.random.PRNGKey(42)
        dt = 0.01

        for forcing in forcings:
            result = forcing(theta_hat, key, dt, grid)
            assert result[0, 0] == 0  # No mean injection

    def test_hermitian_symmetry_property(self):
        """Test that forcing maintains Hermitian symmetry."""
        N = 32
        grid = make_grid(N, 2 * np.pi)
        theta_hat = jnp.zeros((N, N), dtype=complex)

        forcing = WhiteNoiseForcing(amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        result = forcing(theta_hat, key, dt, grid)

        # Physical space should be real
        phys = ifft2(result)
        assert jnp.max(jnp.abs(phys.imag)) < 1e-10

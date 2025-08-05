"""
Tests for ring forcing module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import fft2, ifft2, make_grid
from pygsquig.forcing.ring_forcing import RingForcing, _ensure_hermitian_symmetry


class TestRingForcingInit:
    """Tests for RingForcing initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        forcing = RingForcing(kf=20.0, dk=1.0, epsilon=0.1)

        assert forcing.kf == 20.0
        assert forcing.dk == 1.0
        assert forcing.epsilon == 0.1
        assert forcing.tau_f == 0.0
        assert forcing.amplitude == jnp.sqrt(0.1)

    def test_custom_amplitude(self):
        """Test initialization with custom amplitude."""
        forcing = RingForcing(kf=10.0, dk=2.0, epsilon=0.5, amplitude=2.0)

        assert forcing.amplitude == 2.0

    def test_temporal_correlation(self):
        """Test initialization with temporal correlation."""
        forcing = RingForcing(kf=15.0, tau_f=0.1)

        assert forcing.tau_f == 0.1
        assert forcing.forcing_state is None


class TestForcingMask:
    """Tests for forcing mask generation."""

    def test_mask_ring_selection(self):
        """Test that mask selects correct wavenumbers."""
        grid = make_grid(N=64, L=2 * np.pi)
        forcing = RingForcing(kf=10.0, dk=2.0)

        mask = forcing._get_forcing_mask(grid)

        # Check that mask is boolean
        assert mask.dtype == bool

        # Check modes in the ring
        k_mag = jnp.sqrt(grid.k2)

        # Modes inside ring should be True
        inside_ring = jnp.abs(k_mag - 10.0) <= 1.0
        assert jnp.all(mask[inside_ring])

        # Modes far outside ring should be False
        far_outside = jnp.abs(k_mag - 10.0) > 5.0
        assert jnp.all(~mask[far_outside])

    def test_mask_mode_count(self):
        """Test number of modes in forcing ring."""
        grid = make_grid(N=128, L=2 * np.pi)

        # Narrow ring
        forcing1 = RingForcing(kf=20.0, dk=1.0)
        mask1 = forcing1._get_forcing_mask(grid)
        n_modes1 = jnp.sum(mask1)

        # Wide ring
        forcing2 = RingForcing(kf=20.0, dk=4.0)
        mask2 = forcing2._get_forcing_mask(grid)
        n_modes2 = jnp.sum(mask2)

        # Wider ring should have more modes
        assert n_modes2 > n_modes1

    def test_mask_zero_mode(self):
        """Test that k=0 mode is not forced."""
        grid = make_grid(N=64, L=2 * np.pi)

        # Even with kf close to 0
        forcing = RingForcing(kf=2.0, dk=3.0)
        mask = forcing._get_forcing_mask(grid)

        # Zero mode should not be forced
        assert not mask[0, 0]


class TestRandomPhases:
    """Tests for random phase generation."""

    def test_phase_generation(self):
        """Test random phase generation."""
        forcing = RingForcing(kf=20.0)
        key = jax.random.PRNGKey(42)

        phases = forcing._generate_random_phases(key, shape=(64, 64))

        # Check shape
        assert phases.shape == (64, 64)

        # Check that amplitudes are 1
        amplitudes = jnp.abs(phases)
        np.testing.assert_allclose(amplitudes, 1.0, rtol=1e-10)

        # Check that phases are distributed
        angles = jnp.angle(phases)
        assert angles.min() >= -np.pi
        assert angles.max() <= np.pi

    def test_phase_randomness(self):
        """Test that phases are random with different keys."""
        forcing = RingForcing(kf=20.0)

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        phases1 = forcing._generate_random_phases(key1, shape=(32, 32))
        phases2 = forcing._generate_random_phases(key2, shape=(32, 32))

        # Should be different
        assert not jnp.allclose(phases1, phases2)


class TestEnergyInjection:
    """Tests for energy injection rate control."""

    def test_energy_injection_rate(self):
        """Test that forcing achieves target energy flux."""
        grid = make_grid(N=64, L=2 * np.pi)
        epsilon_target = 0.2
        kf = 15.0
        forcing = RingForcing(kf=kf, dk=2.0, epsilon=epsilon_target)

        # Create a test state with modes that overlap the forcing ring
        # Use a mode close to kf
        k_test = int(kf)
        theta = jnp.sin(k_test * grid.x)
        theta_hat = fft2(theta)

        # Apply forcing
        key = jax.random.PRNGKey(42)
        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)

        # Check energy injection rate
        force_phys = ifft2(force_hat)
        injection_rate = jnp.mean(theta * force_phys)

        # Should match target (approximately, due to randomness)
        # Allow larger tolerance due to randomness
        assert abs(injection_rate - epsilon_target) < 0.5 * epsilon_target

    def test_zero_theta_no_injection(self):
        """Test that zero theta gives zero injection."""
        grid = make_grid(N=64, L=2 * np.pi)
        forcing = RingForcing(kf=20.0, epsilon=0.1)

        # Zero state
        theta_hat = jnp.zeros((64, 64), dtype=jnp.complex128)

        # Apply forcing
        key = jax.random.PRNGKey(42)
        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)

        # With zero theta, can't inject energy
        theta = ifft2(theta_hat)
        force_phys = ifft2(force_hat)
        injection_rate = jnp.mean(theta * force_phys)

        assert injection_rate == 0.0

    def test_consistent_injection_rate(self):
        """Test that injection rate is consistent across calls."""
        grid = make_grid(N=64, L=2 * np.pi)
        epsilon_target = 0.15
        kf = 18.0
        forcing = RingForcing(kf=kf, dk=1.5, epsilon=epsilon_target)

        # State with modes in forcing ring
        k_test = int(kf)
        theta = jnp.sin(k_test * grid.x) + 0.5 * jnp.cos(k_test * grid.y)
        theta_hat = fft2(theta)

        # Multiple forcing applications
        rates = []
        key = jax.random.PRNGKey(42)

        for _i in range(10):
            key, subkey = jax.random.split(key)
            force_hat = forcing(theta_hat, subkey, dt=0.01, grid=grid)

            force_phys = ifft2(force_hat)
            rate = jnp.mean(theta * force_phys)
            rates.append(float(rate))

        # All rates should be close to target
        rates = np.array(rates)
        assert np.all(np.abs(rates - epsilon_target) < 0.2 * epsilon_target)


class TestTemporalCorrelation:
    """Tests for temporal correlation (OU process)."""

    def test_white_noise_no_correlation(self):
        """Test that tau_f=0 gives uncorrelated forcing."""
        grid = make_grid(N=32, L=2 * np.pi)
        forcing = RingForcing(kf=10.0, tau_f=0.0)

        theta_hat = fft2(jnp.ones((32, 32)))

        # Two calls should give uncorrelated results
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        force1 = forcing(theta_hat, key1, dt=0.01, grid=grid)
        force2 = forcing(theta_hat, key2, dt=0.01, grid=grid)

        # Should be different (uncorrelated)
        # Only compute correlation for non-zero modes
        mask = (jnp.abs(force1) > 1e-10) & (jnp.abs(force2) > 1e-10)
        if jnp.sum(mask) > 0:
            correlation = jnp.mean(jnp.conj(force1[mask]) * force2[mask]) / (
                jnp.sqrt(
                    jnp.mean(jnp.abs(force1[mask]) ** 2) * jnp.mean(jnp.abs(force2[mask]) ** 2)
                )
            )
            assert abs(correlation) < 0.2  # Allow some tolerance

    def test_temporal_correlation_decay(self):
        """Test OU process correlation decay."""
        grid = make_grid(N=32, L=2 * np.pi)
        tau_f = 0.1
        forcing = RingForcing(kf=10.0, tau_f=tau_f)

        theta_hat = fft2(jnp.ones((32, 32)))
        dt = 0.01

        # First call initializes state
        key = jax.random.PRNGKey(42)
        force0 = forcing(theta_hat, key, dt=dt, grid=grid)

        # Subsequent calls should show correlation
        correlations = []
        for _i in range(20):
            key, subkey = jax.random.split(key)
            force = forcing(theta_hat, subkey, dt=dt, grid=grid)

            # Compute correlation with initial state
            # Only use non-zero modes
            mask = (jnp.abs(force0) > 1e-10) & (jnp.abs(force) > 1e-10)
            if jnp.sum(mask) > 0:
                norm0 = jnp.sqrt(jnp.mean(jnp.abs(force0[mask]) ** 2))
                norm = jnp.sqrt(jnp.mean(jnp.abs(force[mask]) ** 2))
                corr = jnp.mean(jnp.conj(force0[mask]) * force[mask]).real / (norm0 * norm)
                correlations.append(float(corr))
            else:
                correlations.append(0.0)

        # Correlation should decay exponentially
        correlations = np.array(correlations)
        times = np.arange(len(correlations)) * dt
        expected_decay = np.exp(-times / tau_f)

        # Check first few points (before noise dominates)
        np.testing.assert_allclose(correlations[:5], expected_decay[:5], rtol=0.2)


class TestHermitianSymmetry:
    """Tests for Hermitian symmetry enforcement."""

    def test_ensure_hermitian_dc(self):
        """Test that DC component is made real."""
        field = jnp.ones((64, 64), dtype=jnp.complex128)
        field = field.at[0, 0].set(1.0 + 2.0j)

        field_sym = _ensure_hermitian_symmetry(field)

        # DC should be real
        assert field_sym[0, 0].imag == 0
        assert field_sym[0, 0].real == 1.0

    def test_ensure_hermitian_nyquist(self):
        """Test that Nyquist frequencies are made real."""
        N = 64
        field = jnp.zeros((N, N), dtype=jnp.complex128)

        # Set Nyquist frequencies to complex values
        field = field.at[N // 2, 0].set(1.0 + 1.0j)
        field = field.at[0, N // 2].set(2.0 + 2.0j)
        field = field.at[N // 2, N // 2].set(3.0 + 3.0j)

        field_sym = _ensure_hermitian_symmetry(field)

        # All should be real
        assert field_sym[N // 2, 0].imag == 0
        assert field_sym[0, N // 2].imag == 0
        assert field_sym[N // 2, N // 2].imag == 0

    def test_forcing_produces_real_field(self):
        """Test that forcing produces real physical field."""
        grid = make_grid(N=64, L=2 * np.pi)
        forcing = RingForcing(kf=15.0)

        theta_hat = fft2(jnp.sin(grid.x))
        key = jax.random.PRNGKey(42)

        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)
        force_phys = ifft2(force_hat)

        # Physical field should be real
        np.testing.assert_allclose(force_phys.imag, 0, atol=1e-10)


class TestPowerSpectrum:
    """Tests for forcing power spectrum."""

    def test_power_concentration(self):
        """Test that power is concentrated in forcing ring."""
        grid = make_grid(N=128, L=2 * np.pi)
        kf = 20.0
        dk = 2.0
        forcing = RingForcing(kf=kf, dk=dk)

        theta_hat = fft2(jnp.ones((128, 128)))
        key = jax.random.PRNGKey(42)

        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)

        # Compute power spectrum
        k_mag = jnp.sqrt(grid.k2)
        power = jnp.abs(force_hat) ** 2

        # Power inside ring
        inside_mask = jnp.abs(k_mag - kf) <= dk / 2
        power_inside = jnp.sum(power * inside_mask)

        # Power outside ring (but not too far)
        outside_mask = (jnp.abs(k_mag - kf) > dk / 2) & (k_mag > 0)
        power_outside = jnp.sum(power * outside_mask)

        # Most power should be inside ring
        assert power_inside > 10 * power_outside

    def test_no_forcing_at_zero(self):
        """Test that k=0 mode is not forced."""
        grid = make_grid(N=64, L=2 * np.pi)
        forcing = RingForcing(kf=10.0)

        theta_hat = fft2(jnp.ones((64, 64)))
        key = jax.random.PRNGKey(42)

        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)

        # Zero mode should not be forced
        assert force_hat[0, 0] == 0


class TestDiagnostics:
    """Tests for forcing diagnostics."""

    def test_diagnostics_computation(self):
        """Test diagnostic values."""
        grid = make_grid(N=64, L=2 * np.pi)
        epsilon_target = 0.25
        kf = 16.0
        forcing = RingForcing(kf=kf, dk=1.5, epsilon=epsilon_target)

        # Use theta with modes in forcing ring
        k_test = int(kf)
        theta = jnp.sin(k_test * grid.x)  # Simple mode at kf
        theta_hat = fft2(theta)

        key = jax.random.PRNGKey(42)
        force_hat = forcing(theta_hat, key, dt=0.01, grid=grid)

        diag = forcing.get_diagnostics(theta_hat, force_hat, grid)

        # Check all fields present
        assert "injection_rate" in diag
        assert "forcing_power" in diag
        assert "n_forced_modes" in diag
        assert "target_epsilon" in diag

        # Check values
        assert diag["target_epsilon"] == epsilon_target
        assert diag["n_forced_modes"] > 0
        assert diag["forcing_power"] > 0
        # Injection rate should be close to target when theta overlaps forcing
        assert abs(diag["injection_rate"] - epsilon_target) < 0.5 * epsilon_target


if __name__ == "__main__":
    pytest.main([__file__])

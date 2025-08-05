"""
Tests for deterministic forcing patterns.

This module tests various deterministic forcing implementations
including Taylor-Green, Kolmogorov flow, and other patterns.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pygsquig.core.grid import make_grid, ifft2, fft2
from pygsquig.forcing.deterministic_forcing import (
    TaylorGreenForcing,
    KolmogorovForcing,
    CheckerboardForcing,
    ShearLayerForcing,
    VortexPairForcing,
    TimeModulatedForcing,
    CombinedDeterministicForcing,
    make_taylor_green_forcing,
    make_kolmogorov_forcing,
    make_oscillating_forcing
)
from pygsquig.exceptions import ForcingError


class TestTaylorGreenForcing:
    """Test Taylor-Green vortex forcing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_initialization(self):
        """Test TaylorGreenForcing initialization."""
        forcing = TaylorGreenForcing(amplitude=2.0, k=3)
        assert forcing.amplitude == 2.0
        assert forcing.k == 3
        assert not forcing.time_dependent
        
        # Test validation
        with pytest.raises(ForcingError):
            TaylorGreenForcing(amplitude=-1.0)
        with pytest.raises(ForcingError):
            TaylorGreenForcing(k=0)
            
    def test_steady_forcing(self):
        """Test steady Taylor-Green forcing pattern."""
        forcing = TaylorGreenForcing(amplitude=1.0, k=2)
        
        # Dummy state
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        
        # Compute forcing
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Check pattern at specific points
        # At (0, 0): sin(0)*cos(0) = 0
        assert np.abs(F[0, 0]) < 1e-10
        
        # At (L/4, 0) with k=2: sin(π)*cos(0) = 0
        idx = self.N // 4
        assert np.abs(F[idx, 0]) < 1e-10
        
        # Check that forcing is non-zero overall
        assert jnp.max(jnp.abs(F)) > 0.5
        
    def test_time_dependent_forcing(self):
        """Test time-dependent Taylor-Green forcing."""
        forcing = TaylorGreenForcing(amplitude=1.0, k=2, time_dependent=True)
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        
        # At t=0, cos(0) = 1
        F_hat_0 = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        # Advance time to π/2
        for _ in range(int(np.pi/2 / 0.1)):
            F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        # At t≈π/2, cos(π/2) ≈ 0
        assert jnp.max(jnp.abs(F_hat)) < jnp.max(jnp.abs(F_hat_0)) * 0.1
        
    def test_factory_function(self):
        """Test factory function."""
        forcing = make_taylor_green_forcing(amplitude=2.0, k=3, time_decay=True)
        assert isinstance(forcing, TaylorGreenForcing)
        assert forcing.amplitude == 2.0
        assert forcing.k == 3
        assert forcing.time_dependent


class TestKolmogorovForcing:
    """Test Kolmogorov flow forcing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_y_direction_forcing(self):
        """Test Kolmogorov forcing in y-direction."""
        forcing = KolmogorovForcing(amplitude=1.0, k=4, direction='y')
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Should only depend on y
        # Check that variation along x is minimal (same value in each row)
        for j in range(self.N):
            assert np.std(F[:, j]) < 1e-10
            
        # Check sinusoidal pattern in y
        y_slice = F[:, 0]
        expected = np.sin(4 * 2 * np.pi * self.grid.y[:, 0] / self.L)
        assert np.allclose(y_slice, expected, rtol=1e-5)
        
    def test_x_direction_forcing(self):
        """Test Kolmogorov forcing in x-direction."""
        forcing = KolmogorovForcing(amplitude=2.0, k=3, direction='x')
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Should only depend on x
        # Check that variation along y is minimal (same value in each column)
        for i in range(self.N):
            assert np.std(F[i, :]) < 1e-10
            
        # Check amplitude
        assert np.abs(jnp.max(F) - 2.0) < 1e-5


class TestCheckerboardForcing:
    """Test checkerboard pattern forcing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_square_checkerboard(self):
        """Test square checkerboard pattern."""
        forcing = CheckerboardForcing(amplitude=1.0, kx=4)
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Check that forcing alternates sign
        # The pattern should be sign(sin(kx*x) * sin(ky*y))
        # For k=4, we have 4 full periods in [0, 2π]
        
        # Sample at points where we know the sign
        # Period is L/k = 2π/4
        # At x=π/8, y=π/8: sin(π/2) * sin(π/2) = 1 * 1 = positive
        idx1 = self.N // 16
        assert F[idx1, idx1] > 0
        
        # At x=π/8, y=3π/8: sin(π/2) * sin(3π/2) = 1 * (-1) = negative
        idx2 = 3 * self.N // 16
        assert F[idx1, idx2] < 0
        
        # Check that adjacent cells can have opposite signs
        # Find a transition point
        mid = self.N // 2
        # The pattern should change sign across boundaries
        
    def test_rectangular_checkerboard(self):
        """Test rectangular checkerboard pattern."""
        forcing = CheckerboardForcing(amplitude=2.0, kx=2, ky=4)
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Check amplitude
        assert np.abs(jnp.max(jnp.abs(F)) - 2.0) < 1e-5


class TestShearLayerForcing:
    """Test shear layer forcing patterns."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_linear_shear(self):
        """Test linear shear profile."""
        forcing = ShearLayerForcing(
            amplitude=1.0,
            profile='linear',
            direction='y'
        )
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Should be linear in y
        y_normalized = self.grid.y / self.L - 0.5
        assert np.allclose(F, y_normalized, rtol=1e-5)
        
    def test_tanh_shear(self):
        """Test hyperbolic tangent shear profile."""
        forcing = ShearLayerForcing(
            amplitude=1.0,
            profile='tanh',
            direction='y',
            center=np.pi,
            width=0.5  # Increase width for smoother transition
        )
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Check that we have a transition
        # The forcing should go from negative to positive as y increases
        # Note: y varies along axis 1 (columns), so we check F[0, :]
        
        # At y=0, should be negative (far below center)
        assert F[0, 0] < -0.9
        
        # At y near 2π, should be positive (far above center)
        assert F[0, -1] > 0.9
        
        # At center (y=π), should be near zero
        center_idx = self.N // 2
        assert np.abs(F[0, center_idx]) < 0.1
        
        # Check monotonicity - tanh is strictly increasing
        y_profile = F[0, :]
        diffs = jnp.diff(y_profile)
        assert jnp.all(diffs >= 0)  # Should be non-decreasing
        assert jnp.sum(diffs > 0) > len(diffs) * 0.9  # Mostly increasing


class TestVortexPairForcing:
    """Test vortex pair forcing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_single_vortex(self):
        """Test single Gaussian vortex."""
        vortices = [(np.pi, np.pi, 1.0, 0.5)]  # Center, unit circulation
        forcing = VortexPairForcing(vortices, amplitude_scale=1.0)
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Maximum should be at center
        center_idx = self.N // 2
        assert jnp.argmax(F) == center_idx * self.N + center_idx
        
        # Should decay away from center
        assert F[0, 0] < F[center_idx, center_idx] * 0.1
        
    def test_vortex_dipole(self):
        """Test counter-rotating vortex pair."""
        vortices = [
            (np.pi - 1.0, np.pi, 1.0, 0.3),   # Positive vortex
            (np.pi + 1.0, np.pi, -1.0, 0.3)   # Negative vortex
        ]
        forcing = VortexPairForcing(vortices)
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        F = ifft2(F_hat).real
        
        # Check opposite signs
        left_idx = int(self.N * (np.pi - 1.0) / self.L)
        right_idx = int(self.N * (np.pi + 1.0) / self.L)
        center_idx = self.N // 2
        
        assert F[left_idx, center_idx] * F[right_idx, center_idx] < 0


class TestTimeModulatedForcing:
    """Test time modulation wrapper."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_sine_modulation(self):
        """Test sinusoidal time modulation."""
        base = TaylorGreenForcing(amplitude=1.0, k=2)
        modulated = TimeModulatedForcing(
            base_forcing=base,
            modulation='sine',
            frequency=1.0,
            phase=0.0
        )
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        
        # At t=0, sin(0) = 0
        F_hat_0 = modulated(theta_hat, self.key, dt=0.25, grid=self.grid)
        assert jnp.max(jnp.abs(F_hat_0)) < 1e-10
        
        # At t=0.25, sin(π/2) = 1
        F_hat_1 = modulated(theta_hat, self.key, dt=0.25, grid=self.grid)
        assert jnp.max(jnp.abs(F_hat_1)) > 0.5
        
    def test_exponential_decay(self):
        """Test exponential decay modulation."""
        base = KolmogorovForcing(amplitude=1.0, k=4)
        modulated = TimeModulatedForcing(
            base_forcing=base,
            modulation='exponential',
            decay_rate=2.0
        )
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        
        # Get initial amplitude
        F_hat_0 = modulated(theta_hat, self.key, dt=0.1, grid=self.grid)
        amp_0 = jnp.max(jnp.abs(F_hat_0))
        
        # Advance time
        for _ in range(10):
            F_hat = modulated(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        # Should decay by exp(-2)
        amp_final = jnp.max(jnp.abs(F_hat))
        expected_ratio = np.exp(-2.0)
        assert np.abs(amp_final / amp_0 - expected_ratio) < 0.1
        
    def test_factory_oscillating(self):
        """Test oscillating forcing factory."""
        forcing = make_oscillating_forcing(
            base_pattern='taylor_green',
            frequency=2.0,
            amplitude=3.0,
            k=4
        )
        
        assert isinstance(forcing, TimeModulatedForcing)
        assert isinstance(forcing.base_forcing, TaylorGreenForcing)
        assert forcing.frequency == 2.0
        assert forcing.base_forcing.amplitude == 3.0


class TestCombinedForcing:
    """Test combined deterministic forcing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_linear_combination(self):
        """Test linear combination of forcings."""
        f1 = KolmogorovForcing(amplitude=1.0, k=2, direction='x')
        f2 = KolmogorovForcing(amplitude=1.0, k=2, direction='y')
        
        combined = CombinedDeterministicForcing([
            (1.0, f1),
            (0.5, f2)
        ])
        
        theta_hat = jnp.zeros((self.N, self.N), dtype=complex)
        
        # Get individual forcings
        F1_hat = f1(theta_hat, self.key, dt=0.1, grid=self.grid)
        F2_hat = f2(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        # Get combined forcing
        F_combined_hat = combined(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        # Should be weighted sum
        expected = F1_hat + 0.5 * F2_hat
        assert jnp.allclose(F_combined_hat, expected, rtol=1e-10)
        
    def test_validation(self):
        """Test input validation."""
        with pytest.raises(ForcingError):
            CombinedDeterministicForcing([(1.0, "not a forcing")])


class TestForcingDiagnostics:
    """Test diagnostic computations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.key = jax.random.PRNGKey(42)
        
    def test_energy_injection(self):
        """Test energy injection rate calculation."""
        forcing = TaylorGreenForcing(amplitude=1.0, k=2)
        
        # Create a state that overlaps with forcing
        theta = jnp.sin(2 * 2 * np.pi * self.grid.x / self.L) * \
                jnp.cos(2 * 2 * np.pi * self.grid.y / self.L)
        theta_hat = fft2(theta)
        
        F_hat = forcing(theta_hat, self.key, dt=0.1, grid=self.grid)
        
        diags = forcing.get_diagnostics(theta_hat, F_hat, self.grid)
        
        assert 'injection_rate' in diags
        assert 'forcing_power' in diags
        assert diags['injection_rate'] > 0  # Positive overlap


# Integration test
class TestSolverIntegration:
    """Test integration with solver."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        
    def test_deterministic_forcing_in_solver(self):
        """Test using deterministic forcing in solver."""
        from pygsquig.core.solver import gSQGSolver
        
        # Create solver
        solver = gSQGSolver(
            grid=self.grid,
            alpha=1.0,
            nu_p=1e-4,
            p=8
        )
        
        # Create deterministic forcing
        forcing_obj = KolmogorovForcing(amplitude=0.1, k=4)
        
        # Create wrapper to handle dt parameter
        dt = 0.01
        def forcing_wrapper(theta_hat, **kwargs):
            key = kwargs.get('key', jax.random.PRNGKey(0))
            grid = kwargs.get('grid', self.grid)
            return forcing_obj(theta_hat, key, dt, grid)
        
        # Initialize
        state = solver.initialize(seed=42)
        
        # Step forward with forcing
        key = jax.random.PRNGKey(0)
        new_state = solver.step(state, dt, forcing=forcing_wrapper, key=key, grid=self.grid)
        
        # Should have evolved
        assert new_state['time'] == dt
        assert not jnp.allclose(new_state['theta_hat'], state['theta_hat'])
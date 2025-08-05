"""
Tests for passive scalar evolution module.

This module tests the passive scalar functionality including
advection, diffusion, source terms, and diagnostics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import fft2, ifft2, make_grid
from pygsquig.core.solver_with_scalars import gSQGSolverWithScalars
from pygsquig.scalars.diagnostics import (
    compute_scalar_dissipation,
    compute_scalar_flux,
    compute_scalar_variance,
    compute_scalar_variance_spectrum,
)
from pygsquig.scalars.passive_scalar import (
    MultiSpeciesEvolver,
    PassiveScalarEvolver,
    compute_scalar_advection,
    compute_scalar_diffusion,
)
from pygsquig.scalars.source_terms import (
    ChemicalReaction,
    ExponentialGrowth,
    LocalizedSource,
    TimePeriodicSource,
)
from pygsquig.scalars.state import PassiveScalarState


class TestPassiveScalarBasics:
    """Test basic passive scalar functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_scalar_evolver_creation(self):
        """Test PassiveScalarEvolver initialization."""
        # Basic evolver
        evolver = PassiveScalarEvolver(self.grid, kappa=0.01)
        assert evolver.kappa == 0.01
        assert evolver.source_fn is None
        assert evolver.name == "scalar"

        # With source
        source = ExponentialGrowth(rate=0.1)
        evolver = PassiveScalarEvolver(self.grid, kappa=0.02, source_fn=source, name="tracer")
        assert evolver.source_fn == source
        assert evolver.name == "tracer"

    def test_scalar_initialization(self):
        """Test scalar field initialization."""
        evolver = PassiveScalarEvolver(self.grid)

        # From array
        scalar0 = np.ones((self.N, self.N))
        state = evolver.initialize(scalar0=scalar0)
        assert isinstance(state, PassiveScalarState)
        assert state.scalar_hat.shape == (self.N, self.N)
        assert state.time == 0.0

        # From seed
        state = evolver.initialize(seed=42)
        assert state.scalar_hat.shape == (self.N, self.N)

        # Check smoothing was applied
        k_mag = jnp.sqrt(self.grid.k2)
        high_k_mask = k_mag > self.N // 4 * 2 * np.pi / self.L
        assert jnp.allclose(state.scalar_hat[high_k_mask], 0.0)

    def test_pure_advection(self):
        """Test advection without diffusion or sources."""
        evolver = PassiveScalarEvolver(self.grid, kappa=0.0)

        # Create simple scalar field
        scalar = jnp.sin(2 * self.grid.x) * jnp.cos(2 * self.grid.y)
        state = evolver.initialize(scalar0=scalar)

        # Create uniform velocity
        u = jnp.ones((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        # Step forward
        dt = 0.01
        new_state = evolver.step(state, dt, u, v)

        # Check state updated
        assert new_state.time == dt
        assert not jnp.allclose(new_state.scalar_hat, state.scalar_hat)

    def test_pure_diffusion(self):
        """Test diffusion without advection."""
        kappa = 0.1
        evolver = PassiveScalarEvolver(self.grid, kappa=kappa)

        # High wavenumber initial condition
        k = 4
        scalar = jnp.sin(k * self.grid.x)
        state = evolver.initialize(scalar0=scalar)

        # Zero velocity
        u = jnp.zeros((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        # Step forward
        dt = 0.01
        new_state = evolver.step(state, dt, u, v)

        # Analytical solution: decay rate = kappa * k^2
        expected_decay = np.exp(-kappa * k**2 * dt)

        # Check amplitude decay
        initial_amp = jnp.max(jnp.abs(ifft2(state.scalar_hat)))
        final_amp = jnp.max(jnp.abs(ifft2(new_state.scalar_hat)))
        actual_decay = final_amp / initial_amp

        assert np.isclose(actual_decay, expected_decay, rtol=1e-3)

    def test_conservation_no_source(self):
        """Test scalar conservation without sources or diffusion."""
        evolver = PassiveScalarEvolver(self.grid, kappa=0.0)

        # Random initial condition
        state = evolver.initialize(seed=123)

        # Create divergence-free velocity
        psi = jnp.sin(4 * self.grid.x) * jnp.sin(4 * self.grid.y)
        psi_hat = fft2(psi)
        u = ifft2(-1j * self.grid.ky * psi_hat).real
        v = ifft2(1j * self.grid.kx * psi_hat).real

        # Evolve
        initial_total = jnp.sum(ifft2(state.scalar_hat).real)

        dt = 0.001
        for _ in range(100):
            state = evolver.step(state, dt, u, v)

        final_total = jnp.sum(ifft2(state.scalar_hat).real)

        # Should conserve to machine precision
        assert np.abs(final_total - initial_total) / np.abs(initial_total) < 1e-10


class TestSourceTerms:
    """Test various source term implementations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_exponential_growth(self):
        """Test exponential growth/decay source."""
        # Growth
        rate = 0.5
        source = ExponentialGrowth(rate=rate)
        evolver = PassiveScalarEvolver(self.grid, source_fn=source)

        # Uniform initial condition
        scalar0 = jnp.ones((self.N, self.N))
        state = evolver.initialize(scalar0=scalar0)

        # Zero velocity
        u = v = jnp.zeros((self.N, self.N))

        # Evolve
        dt = 0.01
        t_final = 0.1
        n_steps = int(t_final / dt)

        for _ in range(n_steps):
            state = evolver.step(state, dt, u, v)

        # Check exponential growth
        expected = np.exp(rate * t_final)
        actual = jnp.mean(ifft2(state.scalar_hat).real)
        assert np.isclose(actual, expected, rtol=1e-2)

    def test_localized_source(self):
        """Test Gaussian localized source."""
        source = LocalizedSource(amplitude=1.0, x0=np.pi, y0=np.pi, sigma=0.5)
        evolver = PassiveScalarEvolver(self.grid, source_fn=source)

        # Zero initial condition
        state = evolver.initialize(scalar0=jnp.zeros((self.N, self.N)))

        # Zero velocity
        u = v = jnp.zeros((self.N, self.N))

        # Step forward
        dt = 0.1
        state = evolver.step(state, dt, u, v)

        # Check source created scalar at center
        scalar = ifft2(state.scalar_hat).real
        center_idx = self.N // 2
        assert scalar[center_idx, center_idx] > 0

        # Check Gaussian shape
        assert jnp.argmax(scalar) == center_idx * self.N + center_idx

    def test_chemical_reaction(self):
        """Test quadratic decay source."""
        rate = 1.0
        source = ChemicalReaction(rate=rate)
        evolver = PassiveScalarEvolver(self.grid, source_fn=source)

        # Initial condition with concentration
        scalar0 = jnp.ones((self.N, self.N))
        state = evolver.initialize(scalar0=scalar0)

        # Zero velocity
        u = v = jnp.zeros((self.N, self.N))

        # Small time step for accuracy
        dt = 0.001
        state = evolver.step(state, dt, u, v)

        # For small dt, change ≈ -rate * scalar^2 * dt
        scalar_mean = jnp.mean(ifft2(state.scalar_hat).real)
        expected = 1.0 - rate * 1.0**2 * dt
        assert np.isclose(scalar_mean, expected, rtol=1e-3)

    def test_time_periodic_source(self):
        """Test time-periodic forcing."""
        source = TimePeriodicSource(amplitude=1.0, frequency=2 * np.pi, phase=0.0)  # Period = 1
        evolver = PassiveScalarEvolver(self.grid, source_fn=source)

        # Zero initial condition
        state = evolver.initialize(scalar0=jnp.zeros((self.N, self.N)))
        u = v = jnp.zeros((self.N, self.N))

        # Evolve for quarter period
        dt = 0.01
        for _ in range(25):  # t = 0.25
            state = evolver.step(state, dt, u, v)

        # At t=0.25, sin(2πt) = 1
        scalar_mean = jnp.mean(ifft2(state.scalar_hat).real)
        # Integral of sin from 0 to π/2 is 1
        assert scalar_mean > 0


class TestDiagnostics:
    """Test diagnostic computations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_scalar_variance_spectrum(self):
        """Test variance spectrum computation."""
        # Single mode
        k = 4
        scalar = jnp.sin(k * self.grid.x)
        scalar_hat = fft2(scalar)

        k_bins, spectrum = compute_scalar_variance_spectrum(scalar_hat, self.grid)

        # Should have peak at k=4
        peak_idx = jnp.argmax(spectrum)
        assert np.abs(k_bins[peak_idx] - k) < 2  # Within binning error

    def test_scalar_flux(self):
        """Test turbulent flux computation."""
        # Correlated scalar and velocity
        scalar = jnp.sin(2 * self.grid.x)
        u = jnp.sin(2 * self.grid.x)  # Perfectly correlated
        v = jnp.zeros((self.N, self.N))

        scalar_hat = fft2(scalar)
        flux_x, flux_y = compute_scalar_flux(scalar_hat, u, v, self.grid)

        # Should have positive x-flux
        assert flux_x > 0
        assert np.abs(flux_y) < 1e-10

    def test_scalar_dissipation(self):
        """Test dissipation rate computation."""
        # High wavenumber field
        k = 8
        scalar = jnp.sin(k * self.grid.x)
        scalar_hat = fft2(scalar)

        kappa = 0.1
        chi = compute_scalar_dissipation(scalar_hat, self.grid, kappa)

        # For single mode: χ = κ k² ⟨θ²⟩
        # ⟨sin²(kx)⟩ = 1/2
        expected = kappa * k**2 * 0.5
        # Allow some tolerance due to discretization
        assert np.isclose(chi, expected, rtol=0.05)

    def test_mixing_efficiency(self):
        """Test mixing efficiency computation."""
        # Create initial high-variance field
        k = 4
        scalar_initial = 2.0 * jnp.sin(k * self.grid.x)
        scalar_hat_initial = fft2(scalar_initial)

        # Create final low-variance field (as if diffused)
        scalar_final = 1.0 * jnp.sin(k * self.grid.x)
        scalar_hat_final = fft2(scalar_final)

        time_elapsed = 1.0

        from pygsquig.scalars.diagnostics import compute_mixing_efficiency

        efficiency = compute_mixing_efficiency(
            scalar_hat_initial, scalar_hat_final, self.grid, time_elapsed
        )

        # Should be positive (variance decreases)
        assert efficiency > 0

    def test_scalar_gradient_alignment(self):
        """Test scalar gradient alignment computation."""
        # Create a simple scalar field
        scalar = jnp.sin(2 * self.grid.x) * jnp.cos(2 * self.grid.y)
        scalar_hat = fft2(scalar)

        # Create simple velocity field
        u = jnp.ones((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        from pygsquig.scalars.diagnostics import compute_scalar_gradient_alignment

        alignment = compute_scalar_gradient_alignment(scalar_hat, u, v, self.grid)

        # Check it returns a valid number
        assert isinstance(alignment, float)
        assert not np.isnan(alignment)

    def test_scalar_pdf_moments(self):
        """Test PDF moment computation."""
        # Create a non-Gaussian field
        scalar = jnp.sin(2 * self.grid.x) * jnp.cos(2 * self.grid.y)
        scalar = scalar + 0.1 * scalar**3  # Add some non-Gaussianity
        scalar_hat = fft2(scalar)

        from pygsquig.scalars.diagnostics import compute_scalar_pdf_moments

        moments = compute_scalar_pdf_moments(scalar_hat, max_moment=4)

        # Check all expected moments are present
        assert "mean" in moments
        assert "moment_2" in moments
        assert "moment_3" in moments
        assert "moment_4" in moments
        assert "skewness" in moments
        assert "kurtosis" in moments

        # Variance should be positive
        assert moments["moment_2"] > 0

    def test_batchelor_scale(self):
        """Test Batchelor scale computation."""
        scalar = jnp.sin(4 * self.grid.x)
        scalar_hat = fft2(scalar)

        kappa = 0.01
        epsilon = 1.0  # Turbulent dissipation rate

        from pygsquig.scalars.diagnostics import compute_batchelor_scale

        l_B = compute_batchelor_scale(scalar_hat, self.grid, kappa, epsilon)

        # Check it's positive and finite
        assert l_B > 0
        assert np.isfinite(l_B)

        # Check scaling: l_B ~ (κ³/ε)^(1/4)
        expected = (kappa**3 / epsilon) ** (1 / 4)
        assert np.isclose(l_B, expected)

    def test_peclet_number(self):
        """Test Péclet number computation."""
        u_rms = 1.0
        L_integral = 2.0
        kappa = 0.01

        from pygsquig.scalars.diagnostics import compute_peclet_number

        Pe = compute_peclet_number(u_rms, L_integral, kappa)

        # Check calculation
        expected = u_rms * L_integral / kappa
        assert np.isclose(Pe, expected)

        # Test zero diffusivity
        Pe_inf = compute_peclet_number(u_rms, L_integral, 0.0)
        assert Pe_inf == np.inf


class TestSolverIntegration:
    """Test integration with main solver."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_solver_with_scalars(self):
        """Test extended solver with passive scalars."""
        # Create solver with two scalars
        passive_scalars = {
            "tracer": {"kappa": 0.01},
            "temperature": {"kappa": 0.02, "source": LocalizedSource(1.0, np.pi, np.pi, 0.5)},
        }

        solver = gSQGSolverWithScalars(
            self.grid, alpha=1.0, nu_p=1e-4, p=8, passive_scalars=passive_scalars
        )

        # Initialize
        theta0 = np.random.randn(self.N, self.N)
        scalar_init = {
            "tracer": np.ones((self.N, self.N)),
            "temperature": np.zeros((self.N, self.N)),
        }

        state = solver.initialize(theta0=theta0, scalar_init=scalar_init)

        # Step forward
        dt = 0.01
        new_state = solver.step(state, dt)

        # Check all fields updated
        assert new_state.time == dt
        assert not jnp.allclose(new_state.theta_hat, state.theta_hat)
        assert "tracer" in new_state.scalar_state.scalars
        assert "temperature" in new_state.scalar_state.scalars

    def test_diagnostics_with_scalars(self):
        """Test diagnostic output includes scalars."""
        solver = gSQGSolverWithScalars(
            self.grid, alpha=1.0, passive_scalars={"dye": {"kappa": 0.01}}
        )

        state = solver.initialize(seed=42, scalar_init={"dye": np.ones((self.N, self.N))})

        diags = solver.get_diagnostics(state)

        # Should have both active and passive scalar diagnostics
        assert "kinetic_energy" in diags
        assert "dye_mean" in diags
        assert "dye_variance" in diags

    def test_backward_compatibility(self):
        """Test solver works without scalars."""
        solver = gSQGSolverWithScalars(self.grid, alpha=1.0)

        # Should work like base solver
        state = solver.initialize(seed=42)
        assert isinstance(state, dict)  # Base state

        new_state = solver.step(state, 0.01)
        assert isinstance(new_state, dict)

        diags = solver.get_diagnostics(state)
        assert "kinetic_energy" in diags


# Property-based tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_invalid_diffusivity(self):
        """Test that negative diffusivity raises error."""
        from pygsquig.exceptions import ValidationError

        with pytest.raises(ValidationError):
            PassiveScalarEvolver(self.grid, kappa=-0.1)

    def test_invalid_source_term(self):
        """Test that invalid source term raises error."""
        from pygsquig.exceptions import PassiveScalarError

        with pytest.raises(PassiveScalarError):
            # Pass a non-SourceTerm object
            PassiveScalarEvolver(self.grid, source_fn="not a source term")

    def test_no_initial_condition(self):
        """Test that missing initial condition raises error."""
        from pygsquig.exceptions import PassiveScalarError

        evolver = PassiveScalarEvolver(self.grid)
        with pytest.raises(PassiveScalarError):
            evolver.initialize()  # Neither scalar0 nor seed provided

    def test_invalid_chemical_reaction_rate(self):
        """Test that negative reaction rate raises error."""
        from pygsquig.exceptions import SourceTermError

        with pytest.raises(SourceTermError):
            ChemicalReaction(rate=-1.0)  # Negative rate not allowed

    def test_invalid_localized_source_sigma(self):
        """Test that invalid sigma raises error."""
        from pygsquig.exceptions import SourceTermError

        with pytest.raises(SourceTermError):
            LocalizedSource(amplitude=1.0, x0=0, y0=0, sigma=0.0)  # Zero sigma

        with pytest.raises(SourceTermError):
            LocalizedSource(amplitude=1.0, x0=0, y0=0, sigma=-1.0)  # Negative sigma

    def test_multi_species_empty(self):
        """Test MultiSpeciesEvolver with no species."""
        evolver = MultiSpeciesEvolver(self.grid, species={})
        state = evolver.initialize({})

        # Should have empty scalars
        assert len(state.scalars) == 0

        # Zero velocity
        u = v = jnp.zeros((self.N, self.N))

        # Step should work but do nothing
        new_state = evolver.step(state, 0.01, u, v)
        assert len(new_state.scalars) == 0


class TestProperties:
    """Property-based tests for mathematical invariants."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32  # Smaller for faster tests
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_variance_decay_with_diffusion(self):
        """Test that variance always decreases with diffusion."""
        kappa = 0.1
        evolver = PassiveScalarEvolver(self.grid, kappa=kappa)

        # Random initial condition
        state = evolver.initialize(seed=456)
        u = v = jnp.zeros((self.N, self.N))

        var_initial = compute_scalar_variance(state.scalar_hat)

        # Evolve
        dt = 0.01
        for _ in range(10):
            state = evolver.step(state, dt, u, v)

        var_final = compute_scalar_variance(state.scalar_hat)

        # Variance must decrease
        assert var_final < var_initial

    def test_positivity_preservation(self):
        """Test that positive scalars remain positive (no sources)."""
        evolver = PassiveScalarEvolver(self.grid, kappa=0.01)

        # Positive initial condition
        scalar0 = jnp.ones((self.N, self.N)) + 0.1 * jnp.sin(2 * self.grid.x)
        state = evolver.initialize(scalar0=scalar0)

        # Random divergence-free velocity
        key = jax.random.PRNGKey(789)
        psi_hat = jax.random.normal(key, (self.N, self.N), dtype=jnp.complex128)
        psi_hat = psi_hat * (jnp.sqrt(self.grid.k2) < 10)  # Low-pass filter
        u = ifft2(-1j * self.grid.ky * psi_hat).real
        v = ifft2(1j * self.grid.kx * psi_hat).real

        # Evolve
        dt = 0.001
        for _ in range(100):
            state = evolver.step(state, dt, u, v)
            scalar = ifft2(state.scalar_hat).real
            assert jnp.min(scalar) > -1e-10  # Allow small numerical errors

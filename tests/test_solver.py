"""
Tests for the gSQG solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import fft2, ifft2, make_grid
from pygsquig.core.solver import State, gSQGSolver


class TestSolverInitialization:
    """Tests for solver initialization."""

    def test_solver_creation(self):
        """Test basic solver creation."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-8, p=8)

        assert solver.grid == grid
        assert solver.alpha == 1.0
        assert solver.nu_p == 1e-8
        assert solver.p == 8

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        grid = make_grid(N=64, L=2 * np.pi)

        with pytest.raises(ValueError, match="alpha must be in"):
            gSQGSolver(grid, alpha=3.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            gSQGSolver(grid, alpha=-3.0)

    def test_invalid_p(self):
        """Test that invalid p raises error."""
        grid = make_grid(N=64, L=2 * np.pi)

        with pytest.raises(ValueError, match="p must be 2, 4, or 8"):
            gSQGSolver(grid, alpha=1.0, p=3)

    def test_state_initialization_zero(self):
        """Test zero initial condition."""
        grid = make_grid(N=32, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        state = solver.initialize()

        assert "theta_hat" in state
        assert "time" in state
        assert "step" in state

        assert state["theta_hat"].shape == (32, 32)
        assert jnp.allclose(state["theta_hat"], 0)
        assert state["time"] == 0.0
        assert state["step"] == 0

    def test_state_initialization_custom(self):
        """Test custom initial condition."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        # Custom IC: single mode
        theta0 = jnp.sin(2 * grid.x) * jnp.cos(3 * grid.y)
        state = solver.initialize(theta0=theta0)

        # Check that IC was properly transformed
        theta_back = ifft2(state["theta_hat"])
        np.testing.assert_allclose(theta_back, theta0, rtol=1e-10, atol=1e-15)

    def test_state_initialization_random(self):
        """Test random initial condition."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        state = solver.initialize(seed=42)

        # Should be non-zero
        assert not jnp.allclose(state["theta_hat"], 0)

        # Should be smooth (high wavenumbers killed)
        theta_hat = state["theta_hat"]
        k_mag = jnp.sqrt(grid.k2) / (2 * np.pi / grid.L)
        high_k_power = jnp.abs(theta_hat[k_mag > grid.N // 4]).max()
        assert high_k_power < 1e-10


class TestVelocityComputation:
    """Tests for velocity computation in solver."""

    def test_compute_velocity_sqg(self):
        """Test velocity computation for SQG case (α=1)."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        # Simple test field
        theta = jnp.sin(2 * grid.x) * jnp.sin(2 * grid.y)
        theta_hat = fft2(theta)

        u, v = solver.compute_velocity(theta_hat)

        # Check divergence-free
        u_hat = fft2(u)
        v_hat = fft2(v)
        div_hat = 1j * grid.kx * u_hat + 1j * grid.ky * v_hat
        div = ifft2(div_hat)

        np.testing.assert_allclose(div, 0, atol=1e-10)


class TestDissipation:
    """Tests for hyperviscous dissipation."""

    def test_no_dissipation(self):
        """Test that nu_p=0 gives no dissipation."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.0)

        theta_hat = fft2(jnp.sin(grid.x))
        dissipation = solver.compute_hyperviscosity(theta_hat)

        assert jnp.allclose(dissipation, 0)

    def test_hyperviscosity_p2(self):
        """Test p=2 hyperviscosity (regular viscosity)."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.1, p=2)

        # Single mode
        kx_mode = 3
        theta = jnp.sin(kx_mode * grid.x)
        theta_hat = fft2(theta)

        dissipation_hat = solver.compute_hyperviscosity(theta_hat)

        # Find where the non-zero mode is
        # sin(3x) will have modes at kx=±3, ky=0
        # Due to FFT ordering, positive k is at the beginning
        idx = jnp.where(jnp.abs(theta_hat) > 1e-10)

        # Check that dissipation is applied correctly
        for i, j in zip(idx[0], idx[1]):
            if jnp.abs(theta_hat[i, j]) > 1e-10:
                k2 = grid.k2[i, j]
                expected = -0.1 * k2 * theta_hat[i, j]
                np.testing.assert_allclose(dissipation_hat[i, j], expected, rtol=1e-10)

    def test_hyperviscosity_p8(self):
        """Test p=8 hyperviscosity."""
        grid = make_grid(N=128, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-16, p=8)

        # Test that high wavenumbers are damped more
        theta_hat = jnp.ones((128, 128), dtype=jnp.complex128)
        dissipation_hat = solver.compute_hyperviscosity(theta_hat)

        # High wavenumbers should have larger dissipation
        low_k_diss = jnp.abs(dissipation_hat[0, 1])  # k=1
        high_k_diss = jnp.abs(dissipation_hat[0, 10])  # k=10

        assert high_k_diss > low_k_diss * 10**6  # 10^8 ratio for k^8


class TestRHS:
    """Tests for right-hand side computation."""

    def test_rhs_stationary_state(self):
        """Test that uniform flow gives zero RHS."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.0)

        # Constant theta (uniform)
        theta_hat = jnp.zeros((64, 64), dtype=jnp.complex128)
        theta_hat = theta_hat.at[0, 0].set(1.0)  # DC component only

        rhs = solver.compute_rhs(theta_hat)

        # Should be zero (no gradients, no dissipation)
        np.testing.assert_allclose(rhs, 0, atol=1e-10)

    def test_rhs_energy_conservation(self):
        """Test energy conservation without dissipation."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.0)

        # Use a smooth initial condition to avoid aliasing issues
        theta = jnp.sin(2 * grid.x) * jnp.cos(3 * grid.y)
        theta_hat = fft2(theta)

        # Compute RHS
        rhs = solver.compute_rhs(theta_hat)

        # For inviscid flow, energy should be conserved
        # Check that d/dt ∫θ² = 0
        # This is equivalent to ∫θ(∂θ/∂t) = 0
        theta_rhs = ifft2(rhs)
        energy_tendency = jnp.mean(theta * theta_rhs)

        # Should be zero for inviscid dynamics
        np.testing.assert_allclose(energy_tendency, 0, atol=1e-10)


class TestTimeEvolution:
    """Tests for time stepping."""

    def test_single_step(self):
        """Test single time step."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-8, p=8)

        # Initial condition
        theta0 = jnp.sin(2 * grid.x) * jnp.cos(2 * grid.y)
        state = solver.initialize(theta0=theta0)

        # Take one step
        dt = 0.01
        new_state = solver.step(state, dt)

        # Check state update
        assert new_state["time"] == dt
        assert new_state["step"] == 1
        assert new_state["theta_hat"].shape == state["theta_hat"].shape

        # Should have evolved (not identical)
        assert not jnp.allclose(new_state["theta_hat"], state["theta_hat"])

    def test_multiple_steps(self):
        """Test multiple time steps."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=1e-6, p=4)

        state = solver.initialize(seed=42)
        dt = 0.001

        # Take 10 steps
        for i in range(10):
            state = solver.step(state, dt)

        assert state["time"] == pytest.approx(10 * dt)
        assert state["step"] == 10

    def test_energy_decay_with_dissipation(self):
        """Test that energy decays with dissipation."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.1, p=2)

        # Initial condition with energy
        theta0 = jnp.sin(4 * grid.x) * jnp.cos(4 * grid.y)
        state = solver.initialize(theta0=theta0)

        # Initial energy
        theta = ifft2(state["theta_hat"])
        E0 = jnp.mean(theta**2)

        # Evolve
        dt = 0.01
        for _ in range(100):
            state = solver.step(state, dt)

        # Final energy
        theta = ifft2(state["theta_hat"])
        E1 = jnp.mean(theta**2)

        # Energy should decrease
        assert E1 < E0


class TestDiagnostics:
    """Tests for diagnostic computations."""

    def test_diagnostics_zero_state(self):
        """Test diagnostics for zero state."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        state = solver.initialize()
        diag = solver.get_diagnostics(state)

        assert diag["kinetic_energy"] == 0.0
        assert diag["enstrophy"] == 0.0
        assert diag["max_theta"] == 0.0
        assert diag["time"] == 0.0
        assert diag["step"] == 0

    def test_diagnostics_single_mode(self):
        """Test diagnostics for single mode."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        # Single mode with known properties
        theta0 = jnp.sin(grid.x)
        state = solver.initialize(theta0=theta0)
        diag = solver.get_diagnostics(state)

        # Check enstrophy = 0.5 * <θ²>
        expected_enstrophy = 0.25  # 0.5 * mean(sin²) = 0.5 * 0.5
        assert abs(diag["enstrophy"] - expected_enstrophy) < 1e-10

        # Check max theta
        assert abs(diag["max_theta"] - 1.0) < 1e-10


class TestSolverWithForcingStubs:
    """Test solver with forcing/damping stubs."""

    def test_with_forcing_stub(self):
        """Test solver with simple forcing."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0)

        # Simple forcing that adds energy at large scales
        def forcing(theta_hat, **kwargs):
            force = jnp.zeros_like(theta_hat)
            # Add forcing at k=1
            force = force.at[1, 0].set(0.1)
            force = force.at[0, 1].set(0.1)
            return force

        state = solver.initialize()

        # Step with forcing
        new_state = solver.step(state, dt=0.01, forcing=forcing)

        # Should have non-zero theta after forcing
        theta = ifft2(new_state["theta_hat"])
        assert jnp.abs(theta).max() > 0

    def test_with_damping_stub(self):
        """Test solver with simple damping."""
        grid = make_grid(N=64, L=2 * np.pi)
        solver = gSQGSolver(grid, alpha=1.0, nu_p=0.0)

        # Linear damping
        def damping(theta_hat, **kwargs):
            return -0.1 * theta_hat

        # Start with non-zero state
        theta0 = jnp.sin(grid.x)
        state = solver.initialize(theta0=theta0)
        E0 = jnp.mean(ifft2(state["theta_hat"]) ** 2)

        # Step with damping
        for _ in range(10):
            state = solver.step(state, dt=0.1, damping=damping)

        # Energy should decrease
        E1 = jnp.mean(ifft2(state["theta_hat"]) ** 2)
        assert E1 < E0


if __name__ == "__main__":
    pytest.main([__file__])

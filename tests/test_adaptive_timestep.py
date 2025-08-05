"""
Tests for adaptive timestepping with CFL control.

This module tests the adaptive timestep selection and
stability monitoring functionality.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.adaptive_solver import AdaptivegSQGSolver
from pygsquig.core.adaptive_timestep import (
    AdaptiveTimestepper,
    CFLConfig,
    compute_advection_cfl,
    compute_diffusion_cfl,
    compute_max_velocity,
    compute_timestep,
    estimate_initial_timestep,
)
from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver


class TestCFLConfig:
    """Test CFL configuration validation."""

    def test_default_config(self):
        """Test default configuration is valid."""
        config = CFLConfig()
        assert 0 < config.cfl_safety <= 1
        assert config.dt_min < config.dt_max
        assert config.growth_factor > 1
        assert 0 < config.shrink_factor < 1

    def test_invalid_safety_factor(self):
        """Test invalid safety factor raises error."""
        with pytest.raises(ValueError):
            CFLConfig(cfl_safety=0.0)
        with pytest.raises(ValueError):
            CFLConfig(cfl_safety=1.5)

    def test_invalid_dt_bounds(self):
        """Test invalid dt bounds raise error."""
        with pytest.raises(ValueError):
            CFLConfig(dt_min=1.0, dt_max=0.1)

    def test_invalid_growth_factor(self):
        """Test invalid growth factor raises error."""
        with pytest.raises(ValueError):
            CFLConfig(growth_factor=0.9)


class TestVelocityComputation:
    """Test velocity and CFL computations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.dx = self.L / self.N

    def test_max_velocity_zero(self):
        """Test max velocity for zero field."""
        u = jnp.zeros((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        max_vel = compute_max_velocity(u, v)
        assert max_vel == 0.0

    def test_max_velocity_uniform(self):
        """Test max velocity for uniform flow."""
        u = jnp.ones((self.N, self.N))
        v = jnp.ones((self.N, self.N))

        max_vel = compute_max_velocity(u, v)
        assert np.isclose(max_vel, np.sqrt(2))

    def test_advection_cfl_calculation(self):
        """Test advection CFL calculation."""
        # Uniform flow
        u = jnp.ones((self.N, self.N)) * 2.0
        v = jnp.zeros((self.N, self.N))

        dt_max = compute_advection_cfl(u, v, self.dx)

        # dt_max * max_vel / dx = 1
        # dt_max * 2.0 / dx = 1
        # dt_max = dx / 2.0
        expected = self.dx / 2.0
        assert np.isclose(dt_max, expected)

    def test_diffusion_cfl_standard(self):
        """Test diffusion CFL for standard Laplacian."""
        nu = 0.1
        dt_max = compute_diffusion_cfl(self.grid, nu, p=1)

        # For standard diffusion: dt < 0.25 * dx^2 / nu
        expected = 0.25 * self.dx**2 / nu
        assert np.isclose(dt_max, expected, rtol=0.1)

    def test_diffusion_cfl_hyperviscosity(self):
        """Test diffusion CFL for hyperviscosity."""
        nu = 1e-8
        p = 4
        dt_max = compute_diffusion_cfl(self.grid, nu, p)

        # Should return positive finite value
        assert 0 < dt_max < np.inf

    def test_diffusion_cfl_zero(self):
        """Test diffusion CFL with zero viscosity."""
        dt_max = compute_diffusion_cfl(self.grid, nu=0.0, p=2)
        assert dt_max == np.inf


class TestTimestepComputation:
    """Test adaptive timestep computation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 64
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.config = CFLConfig()

    def test_advection_limited_timestep(self):
        """Test timestep limited by advection."""
        # Fast flow, no diffusion
        u = jnp.ones((self.N, self.N)) * 10.0
        v = jnp.zeros((self.N, self.N))

        dt, diags = compute_timestep(u, v, self.grid, self.config, nu_p=0.0, p=2)

        assert diags["limited_by"] == "advection"
        assert dt <= diags["dt_adv"]
        assert 0 < diags["cfl_adv"] <= self.config.cfl_safety

    def test_diffusion_limited_timestep(self):
        """Test timestep limited by diffusion."""
        # Slow flow, high diffusion
        u = jnp.zeros((self.N, self.N))
        v = jnp.zeros((self.N, self.N))
        u = u.at[0, 0].set(0.1)  # Very small velocity

        dt, diags = compute_timestep(u, v, self.grid, self.config, nu_p=1.0, p=1)  # High diffusion

        assert diags["limited_by"] == "diffusion"
        assert dt <= diags["dt_diff"]

    def test_timestep_bounds(self):
        """Test timestep respects bounds."""
        config = CFLConfig(dt_min=1e-4, dt_max=1e-2)

        # Very fast flow
        u = jnp.ones((self.N, self.N)) * 1000.0
        v = jnp.zeros((self.N, self.N))

        dt, diags = compute_timestep(u, v, self.grid, config)

        assert dt >= config.dt_min
        assert dt <= config.dt_max

    def test_timestep_growth_limit(self):
        """Test timestep growth is limited."""
        config = CFLConfig(growth_factor=1.5)
        current_dt = 0.001

        # Conditions that would allow large timestep
        u = jnp.ones((self.N, self.N)) * 0.1
        v = jnp.zeros((self.N, self.N))

        dt, diags = compute_timestep(u, v, self.grid, config, current_dt=current_dt)

        assert dt <= current_dt * config.growth_factor

    def test_cfl_violation_shrink(self):
        """Test timestep shrinks on CFL violation."""
        config = CFLConfig(shrink_factor=0.5)

        # Start with timestep that violates CFL
        u = jnp.ones((self.N, self.N)) * 10.0
        v = jnp.zeros((self.N, self.N))

        # Get safe timestep
        dt_safe, _ = compute_timestep(u, v, self.grid, config)

        # Use timestep that's too large
        current_dt = dt_safe * 2.0

        dt_new, diags = compute_timestep(u, v, self.grid, config, current_dt=current_dt)

        # Should shrink
        assert dt_new < current_dt


class TestAdaptiveTimestepper:
    """Test AdaptiveTimestepper class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)
        self.config = CFLConfig()
        self.timestepper = AdaptiveTimestepper(self.grid, self.config, verbose=False)

    def test_initialization(self):
        """Test timestepper initialization."""
        assert self.timestepper.n_steps == 0
        assert self.timestepper.n_rejected == 0
        assert len(self.timestepper.dt_history) == 0

    def test_compute_timestep_first(self):
        """Test first timestep computation."""
        state = {"time": 0.0}
        u = jnp.ones((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        dt, diags = self.timestepper.compute_timestep(state, u, v)

        assert dt > 0
        assert "cfl_adv" in diags
        assert len(self.timestepper.dt_history) == 1
        assert self.timestepper.n_steps == 1

    def test_stability_check_stable(self):
        """Test stability check for stable step."""
        theta_before = jnp.ones((self.N, self.N), dtype=complex)
        theta_after = theta_before * 1.1  # Small growth

        state_before = {"theta_hat": theta_before}
        state_after = {"theta_hat": theta_after}

        is_stable, reason = self.timestepper.check_stability(state_before, state_after, 0.01)

        assert is_stable
        assert reason is None

    def test_stability_check_nan(self):
        """Test stability check detects NaN."""
        state_before = {"theta_hat": jnp.ones((self.N, self.N), dtype=complex)}
        state_after = {"theta_hat": jnp.full((self.N, self.N), np.nan, dtype=complex)}

        is_stable, reason = self.timestepper.check_stability(state_before, state_after, 0.01)

        assert not is_stable
        assert "NaN" in reason

    def test_stability_check_excessive_growth(self):
        """Test stability check detects excessive growth."""
        theta_before = jnp.ones((self.N, self.N), dtype=complex)
        theta_after = theta_before * 20.0  # Excessive growth

        state_before = {"theta_hat": theta_before}
        state_after = {"theta_hat": theta_after}

        is_stable, reason = self.timestepper.check_stability(state_before, state_after, 0.01)

        assert not is_stable
        assert "growth" in reason.lower()

    def test_suggest_retry_timestep(self):
        """Test retry timestep suggestion."""
        dt_failed = 0.01
        dt_retry = self.timestepper.suggest_retry_timestep(dt_failed)

        assert dt_retry < dt_failed
        assert dt_retry == dt_failed * self.config.shrink_factor
        assert self.timestepper.n_rejected == 1

    def test_statistics(self):
        """Test statistics computation."""
        # Add some history
        state = {"time": 0.0}
        u = jnp.ones((self.N, self.N))
        v = jnp.zeros((self.N, self.N))

        for _i in range(5):
            self.timestepper.compute_timestep(state, u, v)
            state["time"] += 0.1

        stats = self.timestepper.get_statistics()

        assert stats["n_steps"] == 5
        assert "dt_mean" in stats
        assert "cfl_mean" in stats
        assert stats["efficiency"] == 1.0  # No rejections


class TestInitialTimestep:
    """Test initial timestep estimation."""

    def test_estimate_advection_only(self):
        """Test initial timestep for advection-only problem."""
        grid = make_grid(128, 2 * np.pi)

        dt_est = estimate_initial_timestep(grid, nu_p=0.0, target_cfl=0.5)

        assert dt_est > 0
        assert dt_est < 1.0  # Reasonable bound

    def test_estimate_with_diffusion(self):
        """Test initial timestep with diffusion."""
        grid = make_grid(128, 2 * np.pi)

        dt_est = estimate_initial_timestep(grid, nu_p=0.01, p=2, target_cfl=0.5)

        # Should be smaller than advection-only
        dt_adv_only = estimate_initial_timestep(grid, nu_p=0.0)
        assert dt_est <= dt_adv_only


class TestAdaptiveSolver:
    """Test adaptive solver integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.N = 32
        self.L = 2 * np.pi
        self.grid = make_grid(self.N, self.L)

    def test_solver_creation(self):
        """Test adaptive solver creation."""
        solver = AdaptivegSQGSolver(self.grid, alpha=1.0, nu_p=1e-4, p=4)

        assert solver is not None
        assert hasattr(solver, "timestepper")
        assert solver.current_dt > 0

    def test_single_adaptive_step(self):
        """Test single adaptive step."""
        solver = AdaptivegSQGSolver(self.grid, alpha=1.0, nu_p=1e-4, p=4, verbose=False)

        state = solver.initialize(seed=42)
        new_state, info = solver.step(state)

        assert new_state["time"] > state["time"]
        assert "dt_used" in info
        assert "dt_diagnostics" in info
        assert info["dt_diagnostics"]["cfl_adv"] > 0

    def test_evolve_to_target_time(self):
        """Test evolution to target time."""
        cfl_config = CFLConfig(cfl_safety=0.5, dt_max=0.01)

        solver = AdaptivegSQGSolver(
            self.grid, alpha=1.0, nu_p=1e-3, p=4, cfl_config=cfl_config, verbose=False
        )

        state = solver.initialize(seed=42)
        t_final = 0.1

        results = solver.evolve(state, t_final, save_interval=0.05)

        assert results["final_state"]["time"] >= t_final
        assert len(results["times"]) >= 2  # At least initial and one save
        assert "statistics" in results
        assert results["statistics"]["total_steps"] > 0

    def test_adaptive_vs_fixed_comparison(self):
        """Compare adaptive vs fixed timestep solver."""
        # Create both solvers
        adaptive_solver = AdaptivegSQGSolver(self.grid, alpha=1.0, nu_p=1e-3, p=4, verbose=False)

        fixed_solver = gSQGSolver(self.grid, alpha=1.0, nu_p=1e-3, p=4)

        # Same initial condition
        state_adaptive = adaptive_solver.initialize(seed=123)
        state_fixed = fixed_solver.initialize(seed=123)

        # Evolve both for a short time
        t_final = 0.05

        # Adaptive evolution
        results_adaptive = adaptive_solver.evolve(state_adaptive, t_final)

        # Fixed timestep evolution
        dt_fixed = 0.0001  # Conservative
        n_steps = int(t_final / dt_fixed)

        for _ in range(n_steps):
            state_fixed = fixed_solver.step(state_fixed, dt_fixed)

        # Results should be similar
        theta_adaptive = results_adaptive["final_state"]["theta_hat"]
        theta_fixed = state_fixed["theta_hat"]

        # Allow for some difference due to different timesteps
        rel_error = jnp.linalg.norm(theta_adaptive - theta_fixed) / jnp.linalg.norm(theta_fixed)
        assert rel_error < 0.1  # 10% relative error acceptable

        # Adaptive should be more efficient
        assert results_adaptive["statistics"]["total_steps"] < n_steps


# Property-based tests
class TestProperties:
    """Property-based tests for adaptive timestepping."""

    def test_cfl_safety_respected(self):
        """Test CFL safety factor is always respected."""
        grid = make_grid(64, 2 * np.pi)
        config = CFLConfig(cfl_safety=0.7, target_cfl=0.5)

        # Various velocity magnitudes
        for vel_mag in [0.1, 1.0, 10.0]:
            u = jnp.ones((64, 64)) * vel_mag
            v = jnp.zeros((64, 64))

            dt, diags = compute_timestep(u, v, grid, config)

            # CFL number should not exceed safety factor
            assert diags["cfl_adv"] <= config.cfl_safety + 1e-10

    def test_timestep_monotonic_with_velocity(self):
        """Test timestep decreases with increasing velocity."""
        grid = make_grid(64, 2 * np.pi)
        config = CFLConfig()

        dt_values = []

        for vel_mag in [0.1, 1.0, 10.0, 100.0]:
            u = jnp.ones((64, 64)) * vel_mag
            v = jnp.zeros((64, 64))

            dt, _ = compute_timestep(u, v, grid, config)
            dt_values.append(dt)

        # Timestep should decrease with velocity
        assert all(dt_values[i] >= dt_values[i + 1] for i in range(len(dt_values) - 1))

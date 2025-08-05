"""
Tests for time integration module.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pygsquig.core.time_integrator import (
    rk4_step,
    ssp_rk3_step,
    compute_cfl,
    compute_diffusion_timestep,
    adaptive_timestep,
)


class TestRK4:
    """Tests for RK4 time integrator."""
    
    def test_rk4_linear_ode(self):
        """Test RK4 on linear ODE: dy/dt = -y."""
        # Initial condition
        y0 = jnp.array(1.0)
        
        # RHS function
        def rhs(y):
            return -y
            
        # Time step
        dt = 0.1
        
        # Exact solution after dt
        y_exact = y0 * np.exp(-dt)
        
        # RK4 solution
        y1 = rk4_step(y0, rhs, dt)
        
        # RK4 is 4th order, so error should be O(dt^5)
        np.testing.assert_allclose(y1, y_exact, rtol=1e-4)
        
    def test_rk4_harmonic_oscillator(self):
        """Test RK4 on harmonic oscillator."""
        # State vector [x, v]
        state0 = jnp.array([1.0, 0.0])
        omega = 2.0
        
        def rhs(state):
            x, v = state
            return jnp.array([v, -omega**2 * x])
            
        dt = 0.01
        
        # Evolve for one period
        T = 2 * np.pi / omega
        n_steps = int(T / dt)
        
        state = state0
        for _ in range(n_steps):
            state = rk4_step(state, rhs, dt)
            
        # Should return close to initial condition after one period
        # Allow for some phase error accumulation
        np.testing.assert_allclose(state[0], state0[0], atol=0.01)  # position
        np.testing.assert_allclose(state[1], state0[1], atol=0.01)  # velocity
        
    def test_rk4_order_of_accuracy(self):
        """Test that RK4 has 4th order accuracy."""
        # Simple test problem: dy/dt = -y, y(0) = 1
        # Exact solution: y(t) = exp(-t)
        
        def rhs(y):
            return -y
            
        y0 = jnp.array(1.0)
        t_final = 0.1  # Short time to avoid accumulation
        
        errors = []
        dts = [0.01, 0.005, 0.0025, 0.00125]
        
        for dt in dts:
            n_steps = int(t_final / dt)
            y = y0
            
            for _ in range(n_steps):
                y = rk4_step(y, rhs, dt)
                
            y_exact = jnp.exp(-t_final)
            error = abs(float(y) - float(y_exact))
            errors.append(error)
            
        # Check 4th order convergence
        rates = [np.log(errors[i]/errors[i+1])/np.log(2) for i in range(len(errors)-1)]
        
        # Should see rates close to 4
        assert all(3.9 < rate < 4.1 for rate in rates)  # RK4 is 4th order


class TestSSPRK3:
    """Tests for SSP-RK3 time integrator."""
    
    def test_ssp_rk3_linear_ode(self):
        """Test SSP-RK3 on linear ODE."""
        y0 = jnp.array(1.0)
        
        def rhs(y):
            return -y
            
        dt = 0.1
        y_exact = y0 * np.exp(-dt)
        
        y1 = ssp_rk3_step(y0, rhs, dt)
        
        # SSP-RK3 is 3rd order
        np.testing.assert_allclose(y1, y_exact, rtol=1e-3)
        
    def test_ssp_rk3_tvd_property(self):
        """Test that SSP-RK3 preserves TVD property for linear advection."""
        # This is a simple test - full TVD testing would require
        # implementing advection with flux limiters
        N = 100
        x = jnp.linspace(0, 1, N)
        
        # Step function initial condition
        u0 = jnp.where(jnp.abs(x - 0.5) < 0.1, 1.0, 0.0)
        
        # Simple advection RHS (simplified - not fully upwinded)
        def rhs(u):
            return -jnp.gradient(u)
            
        dt = 0.001
        u1 = ssp_rk3_step(u0, rhs, dt)
        
        # Check that total variation hasn't increased significantly
        tv0 = jnp.sum(jnp.abs(jnp.diff(u0)))
        tv1 = jnp.sum(jnp.abs(jnp.diff(u1)))
        
        assert tv1 <= tv0 * 1.01  # Allow 1% increase due to numerics


class TestCFL:
    """Tests for CFL computation."""
    
    def test_compute_cfl_zero_velocity(self):
        """Test CFL with zero velocity."""
        N = 64
        u = jnp.zeros((N, N))
        v = jnp.zeros((N, N))
        dx = dy = 0.1
        
        dt_cfl = compute_cfl(u, v, dx, dy)
        
        # Should return infinity for zero velocity
        assert jnp.isinf(dt_cfl)
        
    def test_compute_cfl_uniform_flow(self):
        """Test CFL with uniform flow."""
        N = 64
        u = jnp.ones((N, N)) * 2.0  # u = 2
        v = jnp.ones((N, N)) * 1.0  # v = 1
        dx = dy = 0.1
        
        dt_cfl = compute_cfl(u, v, dx, dy, safety_factor=1.0)
        
        # For spectral methods: dt < dx / (pi * u_max)
        expected_dt = min(dx / (np.pi * 2.0), dy / (np.pi * 1.0))
        
        np.testing.assert_allclose(dt_cfl, expected_dt, rtol=1e-10)
        
    def test_compute_cfl_safety_factor(self):
        """Test CFL safety factor."""
        N = 32
        u = jnp.ones((N, N))
        v = jnp.zeros((N, N))
        dx = 0.1
        
        dt_full = compute_cfl(u, v, dx, dx, safety_factor=1.0)
        dt_safe = compute_cfl(u, v, dx, dx, safety_factor=0.5)
        
        np.testing.assert_allclose(dt_safe, 0.5 * dt_full)


class TestDiffusionTimestep:
    """Tests for diffusion timestep computation."""
    
    def test_diffusion_timestep_basic(self):
        """Test diffusion timestep calculation."""
        nu = 0.01
        k2_max = 100.0
        
        dt_diff = compute_diffusion_timestep(nu, k2_max, safety_factor=1.0)
        
        # dt < 2 / (nu * k2_max)
        expected_dt = 2.0 / (nu * k2_max)
        
        np.testing.assert_allclose(dt_diff, expected_dt)
        
    def test_diffusion_timestep_zero_nu(self):
        """Test with zero diffusion."""
        dt_diff = compute_diffusion_timestep(0.0, 100.0)
        assert jnp.isinf(dt_diff)
        
    def test_diffusion_timestep_safety(self):
        """Test diffusion timestep safety factor."""
        nu = 0.1
        k2_max = 50.0
        
        dt_full = compute_diffusion_timestep(nu, k2_max, safety_factor=1.0)
        dt_safe = compute_diffusion_timestep(nu, k2_max, safety_factor=0.8)
        
        np.testing.assert_allclose(dt_safe, 0.8 * dt_full)


class TestAdaptiveTimestep:
    """Tests for adaptive timestep computation."""
    
    def test_adaptive_timestep_cfl_limited(self):
        """Test when CFL is the limiting factor."""
        N = 64
        u = jnp.ones((N, N)) * 10.0  # Fast flow
        v = jnp.zeros((N, N))
        grid_dx = 0.1
        nu = 0.001  # Small diffusion
        k2_max = 10.0
        
        dt = adaptive_timestep(u, v, grid_dx, nu, k2_max, dt_max=1.0)
        
        # Should be CFL limited
        dt_cfl = compute_cfl(u, v, grid_dx, grid_dx, 0.8)
        np.testing.assert_allclose(dt, dt_cfl, rtol=1e-10)
        
    def test_adaptive_timestep_diffusion_limited(self):
        """Test when diffusion is the limiting factor."""
        N = 64
        u = jnp.ones((N, N)) * 0.1  # Slow flow
        v = jnp.zeros((N, N))
        grid_dx = 1.0
        nu = 0.1  # Large diffusion
        k2_max = 1000.0  # High wavenumbers
        
        dt = adaptive_timestep(u, v, grid_dx, nu, k2_max, dt_max=1.0)
        
        # Should be diffusion limited
        dt_diff = compute_diffusion_timestep(nu, k2_max, 0.8)
        np.testing.assert_allclose(dt, dt_diff, rtol=1e-10)
        
    def test_adaptive_timestep_max_limited(self):
        """Test when dt_max is the limiting factor."""
        N = 64
        u = jnp.ones((N, N)) * 0.01  # Very slow flow
        v = jnp.zeros((N, N))
        grid_dx = 1.0
        dt_max = 0.001
        
        dt = adaptive_timestep(u, v, grid_dx, dt_max=dt_max)
        
        # Should be limited by dt_max
        np.testing.assert_allclose(dt, dt_max)


class TestTimeIntegratorStability:
    """Test stability properties of time integrators."""
    
    def test_rk4_energy_conservation(self):
        """Test that RK4 approximately conserves energy for Hamiltonian system."""
        # Simple harmonic oscillator: H = (p^2 + q^2)/2
        
        def rhs(state):
            q, p = state
            return jnp.array([p, -q])
            
        state0 = jnp.array([1.0, 0.0])
        dt = 0.01
        
        # Initial energy
        E0 = 0.5 * (state0[0]**2 + state0[1]**2)
        
        # Evolve for many steps
        state = state0
        for _ in range(1000):
            state = rk4_step(state, rhs, dt)
            
        # Final energy
        E1 = 0.5 * (state[0]**2 + state[1]**2)
        
        # Energy should be conserved to high accuracy
        np.testing.assert_allclose(E1, E0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
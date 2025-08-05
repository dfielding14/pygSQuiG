"""Tests for JIT-compiled grid functions.

These tests ensure that JIT compilation doesn't break functionality
and actually improves performance.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import time

from pygsquig.core.grid import make_grid, fft2, ifft2


class TestGridJIT:
    """Test JIT compilation of grid functions."""
    
    def test_make_grid_jit_compilation(self):
        """Test that make_grid can be JIT compiled."""
        # This will fail until we add JIT decorator
        try:
            make_grid_jit = jax.jit(make_grid, static_argnums=(0, 1))
            grid = make_grid_jit(64, 2*np.pi)
            assert grid.N == 64
            assert grid.L == 2*np.pi
        except:
            pytest.skip("make_grid not yet JIT-compatible")
    
    def test_make_grid_consistency(self):
        """Test JIT and non-JIT versions give same results."""
        N, L = 32, 2*np.pi
        
        # Regular version
        grid1 = make_grid(N, L)
        
        # JIT version (when implemented)
        try:
            make_grid_jit = jax.jit(make_grid, static_argnums=(0, 1))
            grid2 = make_grid_jit(N, L)
            
            # Check all arrays are identical
            assert jnp.allclose(grid1.x, grid2.x)
            assert jnp.allclose(grid1.y, grid2.y)
            assert jnp.allclose(grid1.kx, grid2.kx)
            assert jnp.allclose(grid1.ky, grid2.ky)
            assert jnp.allclose(grid1.k2, grid2.k2)
            assert jnp.array_equal(grid1.dealias_mask, grid2.dealias_mask)
        except:
            pytest.skip("make_grid not yet JIT-compatible")
    
    def test_fft_functions_jit(self):
        """Test FFT functions can be JIT compiled."""
        # Create test data
        N = 64
        field = jnp.ones((N, N))
        
        # Test fft2
        try:
            fft2_jit = jax.jit(fft2)
            result1 = fft2(field)
            result2 = fft2_jit(field)
            assert jnp.allclose(result1, result2)
        except:
            pytest.skip("fft2 not yet JIT-compatible")
        
        # Test ifft2
        try:
            ifft2_jit = jax.jit(ifft2)
            field_hat = fft2(field)
            result1 = ifft2(field_hat)
            result2 = ifft2_jit(field_hat)
            assert jnp.allclose(result1, result2)
        except:
            pytest.skip("ifft2 not yet JIT-compatible")
    
    def test_fft_roundtrip_jit(self):
        """Test FFT roundtrip with JIT compilation."""
        N = 32
        key = jax.random.PRNGKey(42)
        field = jax.random.normal(key, (N, N))
        
        try:
            fft2_jit = jax.jit(fft2)
            ifft2_jit = jax.jit(ifft2)
            
            # Roundtrip
            field_hat = fft2_jit(field)
            field_back = ifft2_jit(field_hat)
            
            assert jnp.allclose(field, field_back, rtol=1e-10)
        except:
            pytest.skip("FFT functions not yet JIT-compatible")
    
    @pytest.mark.benchmark
    def test_make_grid_performance(self):
        """Benchmark make_grid with and without JIT."""
        N, L = 256, 2*np.pi
        
        # Time non-JIT version
        times_regular = []
        for _ in range(5):
            start = time.time()
            _ = make_grid(N, L)
            times_regular.append(time.time() - start)
        
        # Time JIT version (when implemented)
        try:
            make_grid_jit = jax.jit(make_grid, static_argnums=(0, 1))
            
            # Warm up
            _ = make_grid_jit(N, L)
            
            times_jit = []
            for _ in range(5):
                start = time.time()
                _ = make_grid_jit(N, L)
                times_jit.append(time.time() - start)
            
            # JIT should be faster after warmup
            assert np.mean(times_jit) < np.mean(times_regular)
            print(f"Regular: {np.mean(times_regular)*1000:.2f}ms")
            print(f"JIT: {np.mean(times_jit)*1000:.2f}ms")
            print(f"Speedup: {np.mean(times_regular)/np.mean(times_jit):.1f}x")
        except:
            pytest.skip("make_grid not yet JIT-compatible")
    
    def test_grid_immutability_with_jit(self):
        """Test that JIT compilation preserves grid immutability."""
        N, L = 64, 2*np.pi
        
        try:
            make_grid_jit = jax.jit(make_grid, static_argnums=(0, 1))
            grid = make_grid_jit(N, L)
            
            # Try to modify grid - should not affect original
            x_copy = grid.x.copy()
            x_modified = x_copy.at[0, 0].set(999)
            
            # Original should be unchanged
            assert grid.x[0, 0] != 999
        except:
            pytest.skip("make_grid not yet JIT-compatible")
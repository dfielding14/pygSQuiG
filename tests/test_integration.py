"""Integration tests for the full pygSQuiG system.

These tests verify that all components work together correctly.
"""

import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygsquig.core.grid import ifft2, make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.forcing import RingForcing
from pygsquig.forcing.damping import CombinedDamping
from pygsquig.io import (
    GridConfig,
    RunConfig,
    SimulationConfig,
    SolverConfig,
    load_checkpoint,
    save_checkpoint,
    save_diagnostics,
)
from pygsquig.utils import compute_enstrophy, compute_total_energy


class TestBasicSimulation:
    """Test basic simulation functionality."""

    def test_minimal_simulation(self):
        """Test a minimal simulation runs without errors."""
        # Create minimal config
        config = RunConfig(
            grid=GridConfig(N=32, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=0.1),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p,
        )

        # Create initial state
        state = solver.initialize(seed=42)

        # Run a few steps
        dt = 0.001
        for _ in range(10):
            state = solver.step(state, dt)

        # Check state is valid
        assert state["time"] > 0
        assert state["step"] == 10
        assert not jnp.any(jnp.isnan(state["theta_hat"]))

    def test_simulation_with_forcing(self):
        """Test simulation with ring forcing."""
        # Config with forcing
        config = RunConfig(
            grid=GridConfig(N=64, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=0.5),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(grid=grid, alpha=config.solver.alpha)
        state = solver.initialize(seed=123)

        # Create forcing
        forcing = RingForcing(kf=10.0, dk=1.0, epsilon=0.1)
        rng_key = jax.random.PRNGKey(456)

        # Track energy
        initial_energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)

        # Run with forcing
        dt = 0.001
        for i in range(100):
            rng_key, subkey = jax.random.split(rng_key)
            forcing_fn = lambda theta_hat: forcing(theta_hat, subkey, dt, grid)
            state = solver.step(state, dt, forcing=forcing_fn)

        # Energy should increase with forcing
        final_energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)
        assert final_energy > initial_energy

    def test_simulation_with_damping(self):
        """Test simulation with damping."""
        # Config
        config = RunConfig(
            grid=GridConfig(N=64, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=0.5),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p,
        )
        state = solver.initialize(seed=789)

        # Create damping
        damping = CombinedDamping(
            mu=0.1, kf=10.0, nu_p=config.solver.dissipation.nu_p, p=config.solver.dissipation.p
        )

        # Track energy
        initial_energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)

        # Run with damping
        dt = 0.001
        for _ in range(100):
            state = solver.step(state, dt, damping=damping, grid=grid)

        # Energy should decrease with damping
        final_energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)
        assert final_energy < initial_energy


class TestCheckpointing:
    """Test checkpoint/restart functionality."""

    def test_checkpoint_restart(self):
        """Test saving and loading checkpoints."""
        # Create config
        config = RunConfig(
            grid=GridConfig(N=32, L=2 * np.pi),
            solver=SolverConfig(alpha=1.5),
            simulation=SimulationConfig(t_end=1.0),
        )

        # Initialize and run
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(grid=grid, alpha=config.solver.alpha)
        state = solver.initialize(seed=111)

        # Run for a bit
        dt = 0.001
        for _ in range(50):
            state = solver.step(state, dt)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.h5"
            save_checkpoint(state, config, checkpoint_path)

            # Load checkpoint
            loaded_state, loaded_config = load_checkpoint(checkpoint_path)

            # Verify state matches
            assert loaded_state["time"] == state["time"]
            assert loaded_state["step"] == state["step"]
            assert jnp.allclose(loaded_state["theta_hat"], state["theta_hat"])

            # Verify config matches
            assert loaded_config.solver.alpha == config.solver.alpha
            assert loaded_config.grid.N == config.grid.N

    def test_restart_reproducibility(self):
        """Test that restarting gives identical results."""
        # Create config
        config = RunConfig(
            grid=GridConfig(N=32, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=1.0),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(grid=grid, alpha=config.solver.alpha)

        # Run 1: Full simulation
        state1 = solver.initialize(seed=222)
        dt = 0.001
        for _ in range(100):
            state1 = solver.step(state1, dt)

        # Run 2: With checkpoint
        state2 = solver.initialize(seed=222)

        # Run first half
        for _ in range(50):
            state2 = solver.step(state2, dt)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.h5"
            save_checkpoint(state2, config, checkpoint_path)

            # Load and continue
            state2, _ = load_checkpoint(checkpoint_path)
            for _ in range(50):
                state2 = solver.step(state2, dt)

        # Results should be identical
        assert state1["time"] == state2["time"]
        assert state1["step"] == state2["step"]
        assert jnp.allclose(state1["theta_hat"], state2["theta_hat"], rtol=1e-10)


class TestEnergyConservation:
    """Test energy conservation properties."""

    def test_inviscid_energy_conservation(self):
        """Test energy conservation in inviscid case."""
        # Config with no dissipation
        config = RunConfig(
            grid=GridConfig(N=64, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=1.0),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(grid=grid, alpha=config.solver.alpha, nu_p=0.0)
        state = solver.initialize(seed=333)

        # Track energy
        energies = []
        initial_energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)
        energies.append(initial_energy)

        # Run simulation
        dt = 0.0001  # Small timestep for accuracy
        for _ in range(100):
            state = solver.step(state, dt)
            energy = compute_total_energy(state["theta_hat"], grid, config.solver.alpha)
            energies.append(energy)

        # Energy should be approximately conserved
        energies = np.array(energies)
        relative_change = np.abs(energies - initial_energy) / initial_energy
        assert np.max(relative_change) < 1e-3  # Less than 0.1% change

    def test_enstrophy_cascade(self):
        """Test enstrophy behavior in turbulent cascade."""
        # Config with dissipation
        config = RunConfig(
            grid=GridConfig(N=64, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=0.5),
        )

        # Initialize with high amplitude
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p,
        )

        # Create energetic initial condition
        key = jax.random.PRNGKey(444)
        theta_hat = jnp.zeros((grid.N, grid.N), dtype=jnp.complex128)

        # Add energy at intermediate scales
        for kx in range(5, 10):
            for ky in range(5, 10):
                if kx**2 + ky**2 < 100:
                    phase = jax.random.uniform(key, shape=()) * 2 * np.pi
                    theta_hat = theta_hat.at[kx, ky].set(0.1 * jnp.exp(1j * phase))
                    key = jax.random.split(key)[0]

        state = {"theta_hat": theta_hat, "time": 0.0, "step": 0}

        # Track enstrophy
        initial_enstrophy = compute_enstrophy(state["theta_hat"], grid, config.solver.alpha)

        # Run simulation
        dt = 0.001
        for _ in range(100):
            state = solver.step(state, dt)

        # Enstrophy should decrease due to dissipation
        final_enstrophy = compute_enstrophy(state["theta_hat"], grid, config.solver.alpha)
        assert final_enstrophy < initial_enstrophy


class TestDiagnostics:
    """Test diagnostic computations."""

    def test_diagnostics_output(self):
        """Test saving diagnostic time series."""
        # Setup
        config = RunConfig(
            grid=GridConfig(N=32, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=0.1),
        )

        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(grid=grid, alpha=config.solver.alpha)
        state = solver.initialize(seed=555)

        with tempfile.TemporaryDirectory() as tmpdir:
            diag_file = Path(tmpdir) / "diagnostics.h5"

            # Run and save diagnostics
            dt = 0.001
            for i in range(10):
                state = solver.step(state, dt)

                # Compute diagnostics
                diagnostics = {
                    "energy": compute_total_energy(state["theta_hat"], grid, config.solver.alpha),
                    "enstrophy": compute_enstrophy(state["theta_hat"], grid, config.solver.alpha),
                    "theta_max": float(jnp.max(jnp.abs(ifft2(state["theta_hat"])))),
                }

                # Save
                save_diagnostics(diagnostics, state["time"], diag_file)

            # Load and verify
            from pygsquig.io import load_diagnostics

            loaded = load_diagnostics(diag_file)

            assert len(loaded["time"]) == 10
            assert "energy" in loaded
            assert "enstrophy" in loaded
            assert "theta_max" in loaded


class TestScalingBehavior:
    """Test expected scaling behaviors."""

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Requires longer simulation to develop proper cascade")
    def test_sqg_cascade(self):
        """Test SQG develops expected spectral slope."""
        # This is a longer test - mark as slow
        config = RunConfig(
            grid=GridConfig(N=128, L=2 * np.pi),
            solver=SolverConfig(alpha=1.0),
            simulation=SimulationConfig(t_end=10.0),
        )

        # Initialize
        grid = make_grid(config.grid.N, config.grid.L)
        solver = gSQGSolver(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p,
        )

        # Create forcing
        forcing = RingForcing(kf=20.0, dk=2.0, epsilon=0.1)
        rng_key = jax.random.PRNGKey(666)

        # Start from low energy
        state = solver.initialize(seed=777)

        # Run to statistical steady state
        dt = 0.001
        for i in range(5000):
            rng_key, subkey = jax.random.split(rng_key)
            forcing_fn = lambda theta_hat: forcing(theta_hat, subkey, dt, grid)
            state = solver.step(state, dt, forcing=forcing_fn)

        # Compute spectrum
        from pygsquig.utils import compute_energy_spectrum

        k, E_k = compute_energy_spectrum(state["theta_hat"], grid, config.solver.alpha)

        # Check inertial range slope
        # Find inertial range (between forcing and dissipation)
        inertial_mask = (k > 30) & (k < 50)
        k_inertial = k[inertial_mask]
        E_inertial = E_k[inertial_mask]

        # Fit log-log slope
        log_k = np.log(k_inertial)
        log_E = np.log(E_inertial + 1e-20)  # Avoid log(0)
        slope = np.polyfit(log_k, log_E, 1)[0]

        # SQG should have slope close to -5/3
        assert -2.0 < slope < -1.3  # Allow some tolerance

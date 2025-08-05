"""Tests for HDF5 I/O functionality."""

import tempfile
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from pygsquig.core.grid import make_grid
from pygsquig.io import (
    GridConfig,
    RunConfig,
    SolverConfig,
    load_checkpoint,
    load_diagnostics,
    load_output,
    save_checkpoint,
    save_diagnostics,
    save_output,
)


class TestCheckpointIO:
    """Test checkpoint save/load functionality."""

    def test_save_load_basic_checkpoint(self):
        """Test saving and loading a basic checkpoint."""
        # Create test state
        N = 64
        theta_hat = jnp.ones((N, N), dtype=jnp.complex128) * (1 + 2j)
        state = {"theta_hat": theta_hat, "time": 10.5, "step": 1000}

        # Create test config
        config = RunConfig(grid=GridConfig(N=N, L=2 * np.pi), solver=SolverConfig(alpha=1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.h5"

            # Save checkpoint
            save_checkpoint(state, config, checkpoint_path)

            # Check file exists
            assert checkpoint_path.exists()

            # Load checkpoint
            loaded_state, loaded_config = load_checkpoint(checkpoint_path)

            # Check state
            assert jnp.allclose(loaded_state["theta_hat"], state["theta_hat"])
            assert loaded_state["time"] == state["time"]
            assert loaded_state["step"] == state["step"]

            # Check config
            assert loaded_config.grid.N == config.grid.N
            assert loaded_config.grid.L == config.grid.L
            assert loaded_config.solver.alpha == config.solver.alpha

    def test_checkpoint_with_additional_fields(self):
        """Test checkpoint with additional state fields."""
        N = 32
        state = {
            "theta_hat": jnp.ones((N, N), dtype=jnp.complex128),
            "time": 5.0,
            "step": 500,
            "energy": 1.234,  # scalar
            "forcing_state": jnp.array([1.0, 2.0, 3.0]),  # array
            "rng_key": jnp.array([42, 24]),  # JAX random key
        }

        config = RunConfig(grid=GridConfig(N=N), solver=SolverConfig(alpha=1.5))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.h5"

            save_checkpoint(state, config, checkpoint_path)
            loaded_state, _ = load_checkpoint(checkpoint_path)

            # Check all fields
            assert loaded_state["energy"] == state["energy"]
            assert jnp.allclose(loaded_state["forcing_state"], state["forcing_state"])
            assert jnp.array_equal(loaded_state["rng_key"], state["rng_key"])

    def test_checkpoint_compression(self):
        """Test checkpoint compression options."""
        N = 128
        state = {
            "theta_hat": jnp.ones((N, N), dtype=jnp.complex128) * jnp.arange(N * N).reshape(N, N),
            "time": 0.0,
            "step": 0,
        }

        config = RunConfig(grid=GridConfig(N=N), solver=SolverConfig(alpha=1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with different compression options
            path_gzip = Path(tmpdir) / "checkpoint_gzip.h5"
            path_lzf = Path(tmpdir) / "checkpoint_lzf.h5"
            path_none = Path(tmpdir) / "checkpoint_none.h5"

            save_checkpoint(state, config, path_gzip, compression="gzip", compression_level=9)
            save_checkpoint(state, config, path_lzf, compression="lzf")
            save_checkpoint(state, config, path_none, compression=None)

            # Check file sizes (compressed should be smaller)
            size_gzip = path_gzip.stat().st_size
            size_lzf = path_lzf.stat().st_size
            size_none = path_none.stat().st_size

            assert size_gzip < size_none
            assert size_lzf < size_none

            # Verify all can be loaded correctly
            for path in [path_gzip, path_lzf, path_none]:
                loaded_state, _ = load_checkpoint(path)
                assert jnp.allclose(loaded_state["theta_hat"], state["theta_hat"])


class TestOutputIO:
    """Test output save/load functionality."""

    def test_save_load_physical_fields(self):
        """Test saving physical space fields."""
        N = 64
        grid = make_grid(N, 2 * np.pi)

        # Create test fields
        theta = jnp.sin(grid.x) * jnp.cos(2 * grid.y)
        vorticity = -5 * theta  # Fake vorticity

        data = {"theta": theta, "vorticity": vorticity}

        metadata = {"alpha": 1.0, "forcing": True, "nu_p": 1e-8}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.nc"

            # Save output
            save_output(
                data, grid, time=10.0, metadata=metadata, filename=output_path, compress=True
            )

            # Load output
            ds = load_output(output_path)

            # Check data
            assert "theta" in ds
            assert "vorticity" in ds
            assert jnp.allclose(ds["theta"].values, theta)
            assert jnp.allclose(ds["vorticity"].values, vorticity)

            # Check coordinates
            assert jnp.allclose(ds.x.values, grid.x[0, :])
            assert jnp.allclose(ds.y.values, grid.y[:, 0])
            assert ds.time.values == 10.0

            # Check metadata
            assert ds.attrs["alpha"] == 1.0
            assert ds.attrs["forcing"] == "True"  # Booleans converted to strings for NetCDF
            assert ds.attrs["nu_p"] == 1e-8
            assert ds.attrs["N"] == N
            assert ds.attrs["L"] == grid.L

    def test_save_spectral_fields(self):
        """Test saving spectral space fields."""
        N = 32
        grid = make_grid(N, 2 * np.pi)

        # Create test spectral field (rfft2 format)
        theta_hat = jnp.ones((N, N // 2 + 1), dtype=jnp.complex128)

        data = {"theta": theta_hat}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output_spectral.nc"

            save_output(data, grid, time=5.0, metadata={}, filename=output_path)

            ds = load_output(output_path)

            # Should have _hat suffix for spectral fields
            assert "theta_hat" in ds
            assert ds["theta_hat"].shape == (N, N // 2 + 1)

            # Check spectral coordinates
            assert "kx" in ds.coords
            assert "ky" in ds.coords


class TestDiagnosticsIO:
    """Test diagnostics save/load functionality."""

    def test_save_load_scalar_diagnostics(self):
        """Test saving scalar diagnostic quantities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = Path(tmpdir) / "diagnostics.h5"

            # Save initial diagnostics
            diag1 = {"energy": 1.0, "enstrophy": 2.0, "max_vorticity": 0.5}
            save_diagnostics(diag1, time=0.0, filename=diag_path, mode="write")

            # Append more diagnostics
            diag2 = {"energy": 1.1, "enstrophy": 2.2, "max_vorticity": 0.6}
            save_diagnostics(diag2, time=1.0, filename=diag_path, mode="append")

            # Load diagnostics
            loaded = load_diagnostics(diag_path)

            # Check values
            assert len(loaded["time"]) == 2
            assert loaded["time"][0] == 0.0
            assert loaded["time"][1] == 1.0
            assert loaded["energy"][0] == 1.0
            assert loaded["energy"][1] == 1.1
            assert loaded["enstrophy"][0] == 2.0
            assert loaded["enstrophy"][1] == 2.2

    def test_save_load_array_diagnostics(self):
        """Test saving array diagnostic quantities like spectra."""
        k = np.arange(1, 33)

        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = Path(tmpdir) / "diagnostics.h5"

            # Save spectrum at t=0
            spectrum1 = k ** (-5 / 3)
            diag1 = {"energy_spectrum": spectrum1, "total_energy": np.sum(spectrum1)}
            save_diagnostics(diag1, time=0.0, filename=diag_path, mode="write")

            # Save spectrum at t=1
            spectrum2 = 1.1 * k ** (-5 / 3)
            diag2 = {"energy_spectrum": spectrum2, "total_energy": np.sum(spectrum2)}
            save_diagnostics(diag2, time=1.0, filename=diag_path, mode="append")

            # Load diagnostics
            loaded = load_diagnostics(diag_path)

            # Check array values
            assert loaded["energy_spectrum"].shape == (2, 32)
            assert np.allclose(loaded["energy_spectrum"][0], spectrum1)
            assert np.allclose(loaded["energy_spectrum"][1], spectrum2)
            assert loaded["total_energy"][0] == np.sum(spectrum1)
            assert loaded["total_energy"][1] == np.sum(spectrum2)

    def test_append_to_nonexistent_file(self):
        """Test that append mode creates file if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = Path(tmpdir) / "new_diagnostics.h5"

            # This should create the file even though mode="append"
            diag = {"energy": 1.0}
            save_diagnostics(diag, time=0.0, filename=diag_path, mode="append")

            assert diag_path.exists()

            loaded = load_diagnostics(diag_path)
            assert len(loaded["time"]) == 1
            assert loaded["energy"][0] == 1.0

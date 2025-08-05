"""HDF5 I/O functionality for pygSQuiG.

This module provides functions for saving and loading simulation checkpoints
and output data using HDF5 format with xarray integration.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import h5py
import jax.numpy as jnp
import numpy as np
import xarray as xr

from ..core.grid import Grid
from .config import RunConfig


def _get_git_info() -> dict[str, str]:
    """Get current git commit hash and status."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        # Check if there are uncommitted changes
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        is_dirty = len(status) > 0

        return {
            "commit": commit,
            "dirty": is_dirty,
            "status": status[:1000] if is_dirty else "",  # Limit status length
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": False, "status": ""}


def save_checkpoint(
    state: dict[str, Any],
    config: RunConfig,
    filename: Union[str, Path],
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Save simulation checkpoint to HDF5 file.

    Args:
        state: Solver state dictionary containing at minimum:
            - theta_hat: Complex Fourier coefficients
            - time: Current simulation time
            - step: Current step number
        config: Run configuration used for this simulation
        filename: Path to save checkpoint file
        compression: HDF5 compression type ('gzip', 'lzf', or None)
        compression_level: Compression level (1-9 for gzip)
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "w") as f:
        # Save metadata
        meta = f.create_group("metadata")
        meta.attrs["timestamp"] = datetime.now().isoformat()
        meta.attrs["pygsquig_version"] = "0.1.0"  # TODO: Get from package

        # Git information
        git_info = _get_git_info()
        meta.attrs["git_commit"] = git_info["commit"]
        meta.attrs["git_dirty"] = git_info["dirty"]
        if git_info["dirty"]:
            meta.attrs["git_status"] = git_info["status"]

        # Save configuration as JSON
        config_dict = config.to_dict()
        meta.attrs["config"] = json.dumps(config_dict)

        # Save state
        state_group = f.create_group("state")

        # Convert JAX arrays to numpy for storage
        theta_hat = np.array(state["theta_hat"])
        state_group.create_dataset(
            "theta_hat_real",
            data=theta_hat.real,
            compression=compression,
            compression_opts=compression_level if compression == "gzip" else None,
        )
        state_group.create_dataset(
            "theta_hat_imag",
            data=theta_hat.imag,
            compression=compression,
            compression_opts=compression_level if compression == "gzip" else None,
        )

        # Save scalar attributes
        state_group.attrs["time"] = float(state["time"])
        state_group.attrs["step"] = int(state["step"])

        # Save any additional state variables
        for key, value in state.items():
            if key not in ["theta_hat", "time", "step"]:
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    # Array data
                    arr = np.array(value)
                    if np.iscomplexobj(arr):
                        state_group.create_dataset(f"{key}_real", data=arr.real)
                        state_group.create_dataset(f"{key}_imag", data=arr.imag)
                    else:
                        state_group.create_dataset(key, data=arr)
                elif isinstance(value, (int, float, str, bool)):
                    # Scalar data
                    state_group.attrs[key] = value


def load_checkpoint(filename: Union[str, Path]) -> tuple[dict[str, Any], RunConfig]:
    """Load simulation checkpoint from HDF5 file.

    Args:
        filename: Path to checkpoint file

    Returns:
        Tuple of (state_dict, config) where:
            - state_dict: Solver state with JAX arrays
            - config: RunConfig instance
    """
    filename = Path(filename)

    with h5py.File(filename, "r") as f:
        # Load configuration
        config_json = f["metadata"].attrs["config"]
        config_dict = json.loads(config_json)
        config = RunConfig.from_dict(config_dict)

        # Load state
        state_group = f["state"]

        # Reconstruct complex theta_hat
        theta_hat_real = jnp.array(state_group["theta_hat_real"][:])
        theta_hat_imag = jnp.array(state_group["theta_hat_imag"][:])
        theta_hat = theta_hat_real + 1j * theta_hat_imag

        state = {
            "theta_hat": theta_hat,
            "time": float(state_group.attrs["time"]),
            "step": int(state_group.attrs["step"]),
        }

        # Load any additional state variables
        for key in state_group:
            if key not in ["theta_hat_real", "theta_hat_imag"]:
                if key.endswith("_real"):
                    # Complex array
                    base_key = key[:-5]
                    if f"{base_key}_imag" in state_group:
                        real_part = jnp.array(state_group[key][:])
                        imag_part = jnp.array(state_group[f"{base_key}_imag"][:])
                        state[base_key] = real_part + 1j * imag_part
                elif not key.endswith("_imag"):
                    # Real array
                    state[key] = jnp.array(state_group[key][:])

        # Load scalar attributes
        for key, value in state_group.attrs.items():
            if key not in ["time", "step"]:
                state[key] = value

    return state, config


def save_output(
    data: dict[str, jnp.ndarray],
    grid: Grid,
    time: float,
    metadata: dict[str, Any],
    filename: Union[str, Path],
    compress: bool = True,
) -> None:
    """Save simulation output data to HDF5 file using xarray.

    Args:
        data: Dictionary of fields to save (e.g., theta, vorticity)
        grid: Grid object with coordinate information
        time: Current simulation time
        metadata: Additional metadata to save
        filename: Path to save output file
        compress: Whether to use compression
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Create xarray dataset
    data_vars = {}

    for name, field in data.items():
        # Convert JAX arrays to numpy
        field_np = np.array(field)

        # Determine if field is in physical or spectral space
        if field_np.shape == (grid.N, grid.N):
            # Physical space field
            data_vars[name] = xr.DataArray(
                field_np,
                dims=["y", "x"],
                coords={"x": np.array(grid.x[0, :]), "y": np.array(grid.y[:, 0])},
                attrs={"long_name": name, "units": ""},
            )
        elif field_np.shape == (grid.N, grid.N // 2 + 1):
            # Spectral space field (rfft2 format)
            # For rfft2, kx only goes up to N//2+1
            kx_1d = np.array(grid.kx[0, : grid.N // 2 + 1])
            ky_1d = np.array(grid.ky[:, 0])
            data_vars[f"{name}_hat"] = xr.DataArray(
                field_np,
                dims=["ky", "kx"],
                coords={"kx": kx_1d, "ky": ky_1d},
                attrs={"long_name": f"{name} (Fourier space)", "units": ""},
            )

    # Add time coordinate
    coords = {"time": time}

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add metadata as attributes
    # Convert booleans to strings for NetCDF compatibility
    for key, value in metadata.items():
        if isinstance(value, bool):
            ds.attrs[key] = str(value)
        else:
            ds.attrs[key] = value
    ds.attrs["created"] = datetime.now().isoformat()

    # Add grid parameters
    ds.attrs["N"] = grid.N
    ds.attrs["L"] = float(grid.L)

    # Save to file
    encoding = {}
    if compress:
        comp = {"zlib": True, "complevel": 4}
        for var in ds.data_vars:
            encoding[var] = comp

    ds.to_netcdf(filename, encoding=encoding, engine="h5netcdf")


def load_output(filename: Union[str, Path]) -> xr.Dataset:
    """Load simulation output from HDF5/NetCDF file.

    Args:
        filename: Path to output file

    Returns:
        xarray Dataset with simulation data
    """
    return xr.open_dataset(filename, engine="h5netcdf")


def save_diagnostics(
    diagnostics: dict[str, Union[np.ndarray, float]],
    time: float,
    filename: Union[str, Path],
    mode: str = "append",
) -> None:
    """Save diagnostic data to HDF5 file.

    Args:
        diagnostics: Dictionary of diagnostic quantities
        time: Current simulation time
        filename: Path to diagnostics file
        mode: 'append' to add to existing file, 'write' to overwrite
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if mode == "write" or not filename.exists():
        # Create new file
        try:
            with h5py.File(filename, "w-") as f:  # 'w-' fails if file exists
                # Create time dataset
                f.create_dataset("time", data=[time], maxshape=(None,), chunks=True)

                # Create datasets for each diagnostic
                for name, value in diagnostics.items():
                    try:
                        if isinstance(value, (int, float)):
                            # Scalar diagnostic
                            f.create_dataset(name, data=[value], maxshape=(None,), chunks=True)
                        else:
                            # Array diagnostic (e.g., spectrum)
                            arr = np.array(value)
                            (1,) + arr.shape
                            maxshape = (None,) + arr.shape
                            f.create_dataset(
                                name, data=arr[np.newaxis, ...], maxshape=maxshape, chunks=True
                            )
                    except (ValueError, RuntimeError):
                        # Dataset already exists, skip
                        pass
        except OSError:
            # File already exists, append instead
            mode = "append"

    if mode == "append" or filename.exists():
        # Append to existing file
        with h5py.File(filename, "a") as f:
            # Check if we need to append or if time already exists
            time_dset = f["time"]
            existing_times = time_dset[:]

            # Only append if this time isn't already in the file
            if not np.any(np.isclose(existing_times, time)):
                # Append time
                time_dset.resize(time_dset.shape[0] + 1, axis=0)
                time_dset[-1] = time

                # Append diagnostics
                for name, value in diagnostics.items():
                    if name in f:
                        dset = f[name]
                        if isinstance(value, (int, float)):
                            # Scalar
                            dset.resize(dset.shape[0] + 1, axis=0)
                            dset[-1] = value
                        else:
                            # Array
                            arr = np.array(value)
                            dset.resize(dset.shape[0] + 1, axis=0)
                            dset[-1, ...] = arr


def load_diagnostics(filename: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load diagnostic data from HDF5 file.

    Args:
        filename: Path to diagnostics file

    Returns:
        Dictionary with diagnostic arrays
    """
    diagnostics = {}

    with h5py.File(filename, "r") as f:
        for key in f:
            diagnostics[key] = f[key][:]

    return diagnostics

"""
Tests for run.py script with passive scalars.

This module tests the integration of passive scalars
into the main simulation runner script.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import h5py
import xarray as xr
from click.testing import CliRunner

from pygsquig.scripts.run import main
from pygsquig.io.config import RunConfig


@pytest.fixture
def scalar_config_dict():
    """Basic configuration with scalars."""
    return {
        "grid": {"N": 64, "L": 6.283185307179586},
        "solver": {
            "alpha": 1.0,
            "dissipation": {"type": "hyperviscosity", "nu_p": 1e-8, "p": 8},
            "time_integration": {"method": "RK4", "dt": 0.01, "adaptive_cfl": False},
        },
        "scalars": {
            "enabled": True,
            "species": [
                {
                    "name": "tracer",
                    "kappa": 0.01,
                    "initial_condition": "uniform",
                    "initial_params": {"value": 1.0},
                },
                {
                    "name": "dye",
                    "kappa": 0.001,
                    "source": {"type": "exponential", "parameters": {"rate": 0.1}},
                    "initial_condition": "zero",
                },
            ],
        },
        "output": {
            "fields": ["theta", "scalars"],
            "diagnostics": ["energy_spectrum"],
            "compress": False,
        },
        "simulation": {
            "t_end": 0.1,
            "output_interval": 0.05,
            "checkpoint_interval": 0.1,
            "log_interval": 0.05,
        },
        "initial_condition": {"type": "random", "seed": 42},
    }


class TestRunWithScalars:
    """Test run.py script with passive scalars."""

    def test_dry_run_with_scalars(self, scalar_config_dict):
        """Test dry run validates scalar configuration."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write config file
            config_path = Path(tmpdir) / "config.yml"
            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            # Run in dry-run mode
            result = runner.invoke(main, [str(config_path), "--dry-run"])

            assert result.exit_code == 0
            assert "Configuration validated successfully!" in result.output
            assert "Grid: 64x64" in result.output

    def test_scalar_initialization(self, scalar_config_dict):
        """Test that scalars are properly initialized."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            output_dir = Path(tmpdir) / "output"

            # Run simulation very briefly
            scalar_config_dict["simulation"]["t_end"] = 0.01
            scalar_config_dict["simulation"]["output_interval"] = 0.005
            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            # Check that it ran successfully
            assert result.exit_code == 0
            assert "Initialized solver with 2 passive scalar(s)" in result.output

            # Check output files
            assert (output_dir / "fields").exists()
            field_files = list((output_dir / "fields").glob("*.nc"))
            assert len(field_files) >= 1

            # Check that scalar fields are in the output
            ds = xr.open_dataset(field_files[0])
            assert "theta" in ds.variables
            assert "scalar_tracer" in ds.variables
            assert "scalar_dye" in ds.variables
            ds.close()

    def test_scalar_diagnostics(self, scalar_config_dict):
        """Test that scalar diagnostics are computed."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            scalar_config_dict["simulation"]["t_end"] = 0.02
            scalar_config_dict["simulation"]["log_interval"] = 0.01
            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            output_dir = Path(tmpdir) / "output"

            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            assert result.exit_code == 0

            # Check diagnostics file
            diag_file = output_dir / "diagnostics" / "timeseries.h5"
            assert diag_file.exists()

            with h5py.File(diag_file, "r") as f:
                # Check standard diagnostics
                assert "time" in f
                assert "energy" in f

                # Check scalar diagnostics
                assert "tracer_mean" in f
                assert "tracer_variance" in f
                assert "dye_mean" in f
                assert "dye_variance" in f

    def test_scalar_source_evolution(self, scalar_config_dict):
        """Test that scalar sources work correctly."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"

            # Configure with exponential growth source
            scalar_config_dict["scalars"]["species"] = [
                {
                    "name": "growing",
                    "kappa": 0.001,  # Small diffusion for stability
                    "source": {"type": "exponential", "parameters": {"rate": 1.0}},
                    "initial_condition": "uniform",
                    "initial_params": {"value": 1.0},
                }
            ]
            # Use smaller timestep for stability
            scalar_config_dict["solver"]["time_integration"]["dt"] = 0.0001
            scalar_config_dict["solver"]["dissipation"]["nu_p"] = 1e-4  # Stronger dissipation
            scalar_config_dict["simulation"]["t_end"] = 0.01
            scalar_config_dict["simulation"]["output_interval"] = 0.01

            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            output_dir = Path(tmpdir) / "output"

            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            assert result.exit_code == 0

            # Check final output
            field_files = sorted((output_dir / "fields").glob("*.nc"))
            assert len(field_files) >= 1

            ds = xr.open_dataset(field_files[-1])
            final_scalar = ds["scalar_growing"].values
            ds.close()

            # With exponential growth rate of 1.0 and t=0.01,
            # field should have grown by factor of ~exp(0.01) ≈ 1.01
            assert final_scalar.mean() > 1.005
            assert final_scalar.mean() < 1.02  # Not too much growth

    def test_multiple_scalar_species(self, scalar_config_dict):
        """Test simulation with multiple scalar species."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"

            # Add more species
            scalar_config_dict["scalars"]["species"] = [
                {
                    "name": f"species_{i}",
                    "kappa": 0.01 * (i + 1),
                    "initial_condition": "random",
                    "initial_params": {"seed": 100 + i, "amplitude": 0.1},
                }
                for i in range(3)
            ]
            scalar_config_dict["simulation"]["t_end"] = 0.01

            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            output_dir = Path(tmpdir) / "output"

            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            assert result.exit_code == 0
            assert "Initialized solver with 3 passive scalar(s)" in result.output

    def test_no_scalars_backward_compatibility(self):
        """Test that run.py still works without scalars."""
        runner = CliRunner()

        config_dict = {
            "grid": {"N": 64, "L": 6.283185307179586},
            "solver": {
                "alpha": 1.0,
                "dissipation": {"type": "hyperviscosity", "nu_p": 1e-8, "p": 8},
                "time_integration": {"method": "RK4", "dt": 0.01, "adaptive_cfl": False},
            },
            "output": {"fields": ["theta"], "diagnostics": ["energy_spectrum"], "compress": False},
            "simulation": {
                "t_end": 0.01,
                "output_interval": 0.01,
                "checkpoint_interval": 0.1,
                "log_interval": 0.01,
            },
            "initial_condition": {"type": "random", "seed": 42},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            output_dir = Path(tmpdir) / "output"

            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            assert result.exit_code == 0
            assert "passive scalar" not in result.output

            # Check output has no scalar fields
            field_files = list((output_dir / "fields").glob("*.nc"))
            assert len(field_files) >= 1

            ds = xr.open_dataset(field_files[0])
            assert "theta" in ds.variables
            assert not any(var.startswith("scalar_") for var in ds.variables)
            ds.close()

    def test_checkpoint_resume_with_scalars(self, scalar_config_dict):
        """Test checkpoint/resume functionality with scalars."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yml"
            scalar_config_dict["simulation"]["t_end"] = 0.02
            scalar_config_dict["simulation"]["checkpoint_interval"] = 0.01

            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            output_dir = Path(tmpdir) / "output"

            # Run first half
            result = runner.invoke(main, [str(config_path), "--output-dir", str(output_dir)])

            assert result.exit_code == 0

            # Find checkpoint
            checkpoint_files = list((output_dir / "checkpoints").glob("step_*.h5"))
            assert len(checkpoint_files) >= 1
            checkpoint_path = checkpoint_files[0]

            # Extend simulation time
            scalar_config_dict["simulation"]["t_end"] = 0.04
            with open(config_path, "w") as f:
                yaml.dump(scalar_config_dict, f)

            # Resume from checkpoint
            result = runner.invoke(
                main,
                [
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                ],
            )

            assert result.exit_code == 0
            assert "Resuming from" in result.output

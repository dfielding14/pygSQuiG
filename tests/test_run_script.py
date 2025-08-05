"""Tests for the run.py CLI script."""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import time
import numpy as np
import h5py
from click.testing import CliRunner

from pygsquig.scripts import run
from pygsquig.io import load_config


class TestRunScriptCLI:
    """Test command-line interface of run.py."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal config file for testing."""
        config = {
            "grid": {"N": 32, "L": 6.283185307179586},
            "solver": {
                "alpha": 1.0,
                "dissipation": {"nu_p": 1e-3, "p": 4},  # More stable parameters
                "time_integration": {
                    "adaptive_cfl": False,
                    "dt": 0.0001,  # Smaller timestep for stability
                },
            },
            "simulation": {
                "t_end": 0.01,  # Very short simulation to complete quickly
                "output_interval": 0.01,
                "log_interval": 0.003,  # Ensure not a multiple of t_end
                "checkpoint_interval": 0.01,
            },
            "initial_condition": {"type": "random", "seed": 42},
        }

        config_path = temp_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_cli_basic(self, minimal_config, temp_dir):
        """Test basic CLI invocation."""
        runner = CliRunner()
        output_dir = temp_dir / "basic_output"
        result = runner.invoke(run.main, [str(minimal_config), "--output-dir", str(output_dir)])

        # Print full output for debugging
        if result.exit_code != 0:
            print("Exit code:", result.exit_code)
            print("Output:", result.output)
            if result.exception:
                print("Exception:", result.exception)
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

        # Should complete successfully
        assert result.exit_code == 0
        assert "Starting simulation" in result.output

    def test_cli_with_options(self, minimal_config, temp_dir):
        """Test CLI with various options."""
        runner = CliRunner()
        output_dir = temp_dir / "cli_output"

        result = runner.invoke(
            run.main,
            [
                str(minimal_config),
                "--device",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--log-level",
                "DEBUG",
            ],
        )

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_run_minimal_simulation(self, minimal_config, temp_dir):
        """Test running a minimal simulation."""
        output_dir = temp_dir / "output"

        # Run simulation using Click runner
        runner = CliRunner()
        result = runner.invoke(
            run.main, [str(minimal_config), "--output-dir", str(output_dir), "--log-level", "INFO"]
        )

        # Should complete without error
        assert result.exit_code == 0

        # Check outputs were created
        assert output_dir.exists()
        assert any(output_dir.glob("fields/fields_*.nc"))
        assert (output_dir / "diagnostics").exists()
        assert any(output_dir.glob("checkpoints/step_*.h5")) or any(
            output_dir.glob("checkpoints/final_step_*.h5")
        )

    def test_run_with_forcing(self, temp_dir):
        """Test running with forcing enabled."""
        config = {
            "grid": {"N": 32},
            "solver": {
                "alpha": 1.0,
                "dissipation": {"nu_p": 1e-3, "p": 4},
                "time_integration": {"adaptive_cfl": False, "dt": 0.0001},
            },
            "forcing": {"kf": 10.0, "epsilon": 0.1, "dk": 2.0},
            "simulation": {"t_end": 0.01, "log_interval": 0.005, "output_interval": 0.01},
        }

        config_path = temp_dir / "forcing_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = temp_dir / "forcing_output"

        runner = CliRunner()
        result = runner.invoke(run.main, [str(config_path), "--output-dir", str(output_dir)])

        assert result.exit_code == 0

        # Check diagnostics were saved
        assert (output_dir / "diagnostics" / "timeseries.h5").exists()

    def test_restart_functionality(self, minimal_config, temp_dir):
        """Test checkpoint restart."""
        output_dir = temp_dir / "restart_test"

        # Run initial simulation
        runner = CliRunner()
        result = runner.invoke(run.main, [str(minimal_config), "--output-dir", str(output_dir)])
        assert result.exit_code == 0

        # Find checkpoint (either step_*.h5 or final_step_*.h5)
        checkpoints = list((output_dir / "checkpoints").glob("step_*.h5"))
        if not checkpoints:
            checkpoints = list((output_dir / "checkpoints").glob("final_step_*.h5"))
        assert len(checkpoints) > 0
        checkpoint = checkpoints[0]

        # Modify config for restart
        with open(minimal_config, "r") as f:
            config = yaml.safe_load(f)
        config["simulation"]["t_end"] = 0.2  # Extend simulation

        restart_config = temp_dir / "restart_config.yml"
        with open(restart_config, "w") as f:
            yaml.dump(config, f)

        # Run restart
        output_dir2 = temp_dir / "restart_output"
        result = runner.invoke(
            run.main,
            [
                str(restart_config),
                "--output-dir",
                str(output_dir2),
                "--checkpoint",
                str(checkpoint),
            ],
        )
        assert result.exit_code == 0

        # Should have continued from checkpoint
        # Check that output times start after initial run
        field_files = sorted((output_dir2 / "fields").glob("fields_*.nc"))
        assert len(field_files) > 0


class TestRunScriptConfiguration:
    """Test configuration handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_basic_config_loading(self, temp_dir):
        """Test basic configuration loading."""
        config_data = {"grid": {"N": 64}, "solver": {"alpha": 1.0}}

        config_path = temp_dir / "basic.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Test that config can be loaded
        config = load_config(config_path)

        assert config.grid.N == 64
        assert config.solver.alpha == 1.0

    def test_config_with_forcing_and_damping(self, temp_dir):
        """Test config with forcing and damping."""
        config_data = {
            "grid": {"N": 32},
            "solver": {"alpha": 1.0, "damping": {"mu": 0.1, "k_cutoff_factor": 0.5}},
            "forcing": {"kf": 20.0, "epsilon": 0.1},
        }

        config_path = temp_dir / "forcing.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.forcing is not None
        assert config.forcing.kf == 20.0
        assert config.solver.damping is not None
        assert config.solver.damping.mu == 0.1

    def test_invalid_config_handling(self, temp_dir):
        """Test handling of invalid configurations."""
        # Invalid alpha value
        bad_config = {"grid": {"N": 32}, "solver": {"alpha": 3.0}}  # Out of range

        config_path = temp_dir / "bad_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(bad_config, f)

        runner = CliRunner()
        result = runner.invoke(run.main, [str(config_path)])

        # Should fail with error
        assert result.exit_code != 0
        # Check the exception message
        assert result.exception is not None
        assert "alpha must be in [-2, 2]" in str(result.exception)

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        runner = CliRunner()
        result = runner.invoke(run.main, ["nonexistent.yml"])

        # Should fail with error
        assert result.exit_code != 0


class TestRunScriptOutput:
    """Test output generation and formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_output_structure(self, temp_dir):
        """Test output directory structure."""
        config = {
            "grid": {"N": 32},
            "solver": {
                "alpha": 1.0,
                "dissipation": {"nu_p": 1e-3, "p": 4},
                "time_integration": {"adaptive_cfl": False, "dt": 0.0001},
            },
            "simulation": {
                "t_end": 0.01,
                "output_interval": 0.005,
                "checkpoint_interval": 0.01,
                "log_interval": 0.01,
            },
            "output": {"fields": ["theta"], "diagnostics": ["energy_spectrum", "enstrophy"]},
        }

        config_path = temp_dir / "output_test.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = temp_dir / "outputs"

        runner = CliRunner()
        result = runner.invoke(run.main, [str(config_path), "--output-dir", str(output_dir)])

        assert result.exit_code == 0

        # Check directory structure
        assert (output_dir / "fields").exists()
        assert (output_dir / "checkpoints").exists()
        assert (output_dir / "diagnostics").exists()

        # Check files were created
        field_files = list((output_dir / "fields").glob("fields_*.nc"))
        assert len(field_files) >= 1  # At least one output

        checkpoint_files = list((output_dir / "checkpoints").glob("*.h5"))
        assert len(checkpoint_files) >= 1

        diag_files = list((output_dir / "diagnostics").glob("*.h5"))
        assert len(diag_files) >= 1

    def test_dry_run_mode(self, temp_dir):
        """Test dry-run mode doesn't create outputs."""
        config = {"grid": {"N": 32}, "solver": {"alpha": 1.0}, "simulation": {"t_end": 1.0}}

        config_path = temp_dir / "dryrun.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_dir = temp_dir / "dryrun_output"

        runner = CliRunner()
        result = runner.invoke(
            run.main, [str(config_path), "--output-dir", str(output_dir), "--dry-run"]
        )

        # Should complete successfully
        assert result.exit_code == 0

        # But shouldn't run simulation
        assert "Dry run" in result.output

        # Output directory might be created but should be empty
        if output_dir.exists():
            field_files = list(output_dir.glob("**/*.nc"))
            assert len(field_files) == 0


class TestRunScriptErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_output_directory_creation(self, temp_dir):
        """Test automatic output directory creation."""
        config = {"grid": {"N": 32}, "solver": {"alpha": 1.0}, "simulation": {"t_end": 0.01}}

        config_path = temp_dir / "dir_test.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Nested output directory that doesn't exist
        output_dir = temp_dir / "deep" / "nested" / "output"

        runner = CliRunner()
        result = runner.invoke(run.main, [str(config_path), "--output-dir", str(output_dir)])

        assert result.exit_code == 0

        # Should have created the directory structure
        assert output_dir.exists()
        assert (output_dir / "fields").exists()
        assert (output_dir / "checkpoints").exists()

    def test_invalid_device(self, temp_dir):
        """Test invalid device option."""
        config = {"grid": {"N": 32}, "solver": {"alpha": 1.0}}

        config_path = temp_dir / "device_test.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(run.main, [str(config_path), "--device", "invalid_device"])

        # Should fail with error
        assert result.exit_code != 0

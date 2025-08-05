"""Main script to run pygSQuiG simulations from YAML configuration files."""

import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np

from pygsquig.core.grid import make_grid
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.time_integrator import adaptive_timestep
from pygsquig.io import (
    load_config,
    save_checkpoint,
    save_diagnostics,
    save_output,
)
from pygsquig.utils import compute_enstrophy, compute_total_energy, setup_logging

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global shutdown_requested
    if hasattr(signal_handler, "logger"):
        signal_handler.logger.warning("Shutdown requested. Saving checkpoint before exiting...")
    shutdown_requested = True


def setup_output_directories(output_dir: Path, config) -> dict:
    """Create output directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    dirs = {
        "checkpoints": output_dir / "checkpoints",
        "fields": output_dir / "fields",
        "diagnostics": output_dir / "diagnostics",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    # Save configuration for reference
    config.to_yaml(output_dir / "config.yml")

    return dirs


def initialize_simulation(config, checkpoint_path=None):
    """Initialize simulation from config or checkpoint."""
    # Create grid
    grid = make_grid(config.grid.N, config.grid.L)

    # Create solver
    solver = gSQGSolver(
        grid=grid,
        alpha=config.solver.alpha,
        nu_p=config.solver.dissipation.nu_p,
        p=config.solver.dissipation.p,
    )

    if checkpoint_path:
        # Load from checkpoint
        click.echo(f"Loading checkpoint: {checkpoint_path}")
        from pygsquig.io import load_checkpoint

        state, loaded_config = load_checkpoint(checkpoint_path)
        click.echo(f"Resuming from t={state['time']:.2f}, step={state['step']}")
    else:
        # Initialize new simulation
        seed = config.initial_condition.seed
        if config.initial_condition.type == "random":
            state = solver.initialize(seed=seed)
        else:
            # TODO: Support other initial condition types
            raise NotImplementedError(
                f"Initial condition type '{config.initial_condition.type}' not implemented"
            )

    return grid, solver, state


def compute_diagnostics(state, grid, solver):
    """Compute diagnostic quantities."""
    # Get theta in physical space
    theta = grid.ifft2(state["theta_hat"])

    # Compute basic diagnostics
    diagnostics = {
        "theta_rms": float(jnp.sqrt(jnp.mean(theta**2))),
        "theta_max": float(jnp.max(jnp.abs(theta))),
    }

    # Add solver diagnostics if available
    solver_diags = solver.compute_diagnostics(state)
    diagnostics.update(solver_diags)

    return diagnostics


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--device", type=click.Choice(["cpu", "gpu", "tpu"]), default="cpu", help="Device to run on"
)
@click.option(
    "--checkpoint", type=click.Path(exists=True), help="Path to checkpoint file to resume from"
)
@click.option(
    "--output-dir", type=click.Path(), default="./output", help="Directory for output files"
)
@click.option("--dry-run", is_flag=True, help="Validate configuration without running")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Console logging level",
)
def main(config, device, checkpoint, output_dir, dry_run, log_level):
    """Run pygSQuiG simulation from YAML configuration file.

    Example:
        pygsquig-run config.yml --device=gpu
        pygsquig-run config.yml --checkpoint=output/checkpoints/step_1000.h5
    """
    # Set up device
    if device == "gpu":
        if not jax.devices("gpu"):
            click.echo("Warning: No GPU found, falling back to CPU", err=True)
            device = "cpu"
        else:
            click.echo(f"Using GPU: {jax.devices('gpu')[0]}")
    elif device == "tpu":
        if not jax.devices("tpu"):
            click.echo("Warning: No TPU found, falling back to CPU", err=True)
            device = "cpu"

    # Load configuration
    click.echo(f"Loading configuration: {config}")
    run_config = load_config(config)

    if dry_run:
        click.echo("Configuration validated successfully!")
        click.echo(f"Grid: {run_config.grid.N}x{run_config.grid.N}, L={run_config.grid.L}")
        click.echo(f"Solver: α={run_config.solver.alpha}")
        click.echo(f"Simulation time: {run_config.simulation.t_end}")
        return

    # Set up output directories
    output_dir = Path(output_dir)
    dirs = setup_output_directories(output_dir, run_config)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize simulation
    click.echo("Initializing simulation...")
    grid, solver, state = initialize_simulation(run_config, checkpoint)

    # Set up timing
    t_start = time.time()
    t_last_output = state["time"]
    t_last_checkpoint = state["time"]
    t_last_log = state["time"]

    # Main time loop
    click.echo(
        f"\nStarting simulation from t={state['time']:.2f} to t={run_config.simulation.t_end:.2f}"
    )
    click.echo("Press Ctrl+C to save and exit gracefully\n")

    # Initialize forcing if configured
    forcing_fn = None
    if run_config.forcing and run_config.forcing.type != "none":
        # TODO: Initialize forcing based on config
        click.echo("Warning: Forcing not yet implemented")

    # Initialize damping if configured
    damping_fn = None
    if run_config.solver.damping and run_config.solver.damping.type != "none":
        # TODO: Initialize damping based on config
        click.echo("Warning: Damping not yet implemented")

    while state["time"] < run_config.simulation.t_end and not shutdown_requested:
        # Compute time step
        if run_config.solver.time_integration.adaptive_cfl:
            from pygsquig.core.operators import compute_velocity_from_theta

            u, v = compute_velocity_from_theta(state["theta_hat"], grid, run_config.solver.alpha)
            dt = adaptive_timestep(
                u,
                v,
                grid.x[1, 0] - grid.x[0, 0],
                grid.y[0, 1] - grid.y[0, 0],
                run_config.solver.time_integration.cfl_safety,
                run_config.solver.time_integration.dt_max,
            )
        else:
            dt = run_config.solver.time_integration.dt

        # Don't overshoot end time
        dt = min(dt, run_config.simulation.t_end - state["time"])

        # Time step
        state = solver.step(state, dt, forcing=forcing_fn, damping=damping_fn)

        # Logging
        if state["time"] - t_last_log >= run_config.simulation.log_interval:
            diagnostics = compute_diagnostics(state, grid, solver)
            elapsed = time.time() - t_start
            eta = (
                elapsed
                * (run_config.simulation.t_end - state["time"])
                / (state["time"] - t_last_log)
                if state["time"] > t_last_log
                else 0
            )

            click.echo(
                f"t={state['time']:8.3f} | step={state['step']:6d} | "
                f"dt={dt:.2e} | θ_rms={diagnostics['theta_rms']:.3e} | "
                f"θ_max={diagnostics['theta_max']:.3e} | "
                f"ETA: {timedelta(seconds=int(eta))}"
            )

            # Save diagnostics
            save_diagnostics(diagnostics, state["time"], dirs["diagnostics"] / "timeseries.h5")

            t_last_log = state["time"]

        # Output fields
        if state["time"] - t_last_output >= run_config.simulation.output_interval:
            # Prepare fields to save
            data = {}
            theta = grid.ifft2(state["theta_hat"])

            for field in run_config.output.fields:
                if field == "theta":
                    data["theta"] = theta
                elif field == "vorticity":
                    # TODO: Implement vorticity calculation
                    # For gSQG, vorticity = (-Δ)^(α/2) θ
                    click.echo("Warning: Vorticity output not yet implemented")
                    continue
                elif field == "streamfunction":
                    from pygsquig.core.operators import compute_streamfunction

                    psi_hat = compute_streamfunction(
                        state["theta_hat"], grid, run_config.solver.alpha
                    )
                    data["streamfunction"] = grid.ifft2(psi_hat)
                elif field == "velocity":
                    from pygsquig.core.operators import compute_velocity_from_theta

                    u, v = compute_velocity_from_theta(
                        state["theta_hat"], grid, run_config.solver.alpha
                    )
                    data["u"] = u
                    data["v"] = v

            # Save output
            output_file = dirs["fields"] / f"fields_{state['step']:08d}.nc"
            metadata = {
                "step": state["step"],
                "alpha": run_config.solver.alpha,
                "nu_p": run_config.solver.dissipation.nu_p,
                "p": run_config.solver.dissipation.p,
            }
            save_output(
                data,
                grid,
                state["time"],
                metadata,
                output_file,
                compress=run_config.output.compress,
            )

            t_last_output = state["time"]

        # Checkpointing
        if state["time"] - t_last_checkpoint >= run_config.simulation.checkpoint_interval:
            checkpoint_file = dirs["checkpoints"] / f"step_{state['step']:08d}.h5"
            save_checkpoint(state, run_config, checkpoint_file)
            click.echo(f"Checkpoint saved: {checkpoint_file}")
            t_last_checkpoint = state["time"]

        # Check wall time limit
        if run_config.simulation.wall_time_limit:
            if time.time() - t_start > run_config.simulation.wall_time_limit:
                click.echo("\nWall time limit reached!")
                shutdown_requested = True

    # Final checkpoint and output
    click.echo("\nSaving final state...")

    # Save final checkpoint
    checkpoint_file = dirs["checkpoints"] / f"final_step_{state['step']:08d}.h5"
    save_checkpoint(state, run_config, checkpoint_file)

    # Save final output if needed
    if state["time"] - t_last_output > 0:
        data = {"theta": grid.ifft2(state["theta_hat"])}
        output_file = dirs["fields"] / f"fields_final_{state['step']:08d}.nc"
        save_output(data, grid, state["time"], {"step": state["step"]}, output_file)

    # Summary
    elapsed = time.time() - t_start
    click.echo(f"\nSimulation completed!")
    click.echo(f"Final time: {state['time']:.3f}")
    click.echo(f"Total steps: {state['step']}")
    click.echo(f"Wall time: {timedelta(seconds=int(elapsed))}")
    click.echo(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

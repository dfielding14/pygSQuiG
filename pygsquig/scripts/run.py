"""Main script to run pygSQuiG simulations from YAML configuration files."""

import click
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime, timedelta
import signal

from pygsquig.io import (
    load_config,
    save_checkpoint,
    save_output,
    save_diagnostics,
)
from pygsquig.core.grid import make_grid, ifft2
from pygsquig.core.solver import gSQGSolver
from pygsquig.core.solver_with_scalars import gSQGSolverWithScalars
from pygsquig.core.time_integrator import adaptive_timestep
from pygsquig.utils import setup_logging, compute_total_energy, compute_enstrophy
from pygsquig.forcing import RingForcing, CombinedDamping
from pygsquig.scalars.source_terms import (
    ExponentialGrowth, LocalizedSource, ChemicalReaction, TimePeriodicSource
)


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True
    if hasattr(signal_handler, 'logger'):
        signal_handler.logger.warning("Shutdown requested. Saving checkpoint before exiting...")


def setup_output_directories(output_dir: Path, config, logger) -> dict:
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
    logger.info(f"Configuration saved to {output_dir / 'config.yml'}")
    
    return dirs


def create_source_from_config(source_config):
    """Create source term instance from configuration."""
    if source_config is None:
        return None
        
    source_type = source_config.type
    params = source_config.parameters
    
    if source_type == "exponential":
        return ExponentialGrowth(rate=params.get("rate", 0.0))
    elif source_type == "localized":
        return LocalizedSource(
            amplitude=params.get("amplitude", 1.0),
            x0=params.get("x0", 0.0),
            y0=params.get("y0", 0.0),
            sigma=params.get("sigma", 1.0)
        )
    elif source_type == "chemical":
        return ChemicalReaction(
            rate=params.get("rate", 1.0),
            threshold=params.get("threshold")
        )
    elif source_type == "periodic":
        return TimePeriodicSource(
            amplitude=params.get("amplitude", 1.0),
            frequency=params.get("frequency", 1.0),
            phase=params.get("phase", 0.0)
        )
    elif source_type == "none":
        return None
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def initialize_simulation(config, checkpoint_path, logger):
    """Initialize simulation from config or checkpoint."""
    # Create grid
    grid = make_grid(config.grid.N, config.grid.L)
    
    # Check if we need scalars
    if config.scalars and config.scalars.enabled:
        # Create solver with scalars
        passive_scalars = {}
        for species in config.scalars.species:
            passive_scalars[species.name] = {
                'kappa': species.kappa,
                'source': create_source_from_config(species.source)
            }
        
        solver = gSQGSolverWithScalars(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p,
            passive_scalars=passive_scalars
        )
        logger.info(f"Initialized solver with {len(passive_scalars)} passive scalar(s)")
    else:
        # Create standard solver
        solver = gSQGSolver(
            grid=grid,
            alpha=config.solver.alpha,
            nu_p=config.solver.dissipation.nu_p,
            p=config.solver.dissipation.p
        )
    
    if checkpoint_path:
        # Load from checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        from pygsquig.io import load_checkpoint
        state, loaded_config = load_checkpoint(checkpoint_path)
        logger.info(f"Resuming from t={state['time']:.2f}, step={state['step']}")
    else:
        # Initialize new simulation
        seed = config.initial_condition.seed
        if config.initial_condition.type == "random":
            # Initialize active scalar
            if config.scalars and config.scalars.enabled:
                # Initialize with scalars
                scalar_init = {}
                for species in config.scalars.species:
                    if species.initial_condition == "zero":
                        scalar_init[species.name] = jnp.zeros((grid.N, grid.N))
                    elif species.initial_condition == "random":
                        key = jax.random.PRNGKey(species.initial_params.get("seed", seed + 1))
                        scalar_init[species.name] = species.initial_params.get("amplitude", 1.0) * \
                                                   jax.random.normal(key, (grid.N, grid.N))
                    elif species.initial_condition == "uniform":
                        value = species.initial_params.get("value", 1.0)
                        scalar_init[species.name] = value * jnp.ones((grid.N, grid.N))
                    elif species.initial_condition == "gaussian":
                        x0, y0 = species.initial_params.get("center", [grid.L/2, grid.L/2])
                        width = species.initial_params.get("width", 1.0)
                        r2 = (grid.x - x0)**2 + (grid.y - y0)**2
                        scalar_init[species.name] = jnp.exp(-r2 / (2 * width**2))
                    else:
                        logger.warning(f"Unknown initial condition '{species.initial_condition}' for {species.name}, using zero")
                        scalar_init[species.name] = jnp.zeros((grid.N, grid.N))
                
                state = solver.initialize(seed=seed, scalar_init=scalar_init)
                logger.info(f"Initialized with random initial condition and {len(scalar_init)} scalar(s) (seed={seed})")
            else:
                state = solver.initialize(seed=seed)
                logger.info(f"Initialized with random initial condition (seed={seed})")
        else:
            # TODO: Support other initial condition types
            raise NotImplementedError(f"Initial condition type '{config.initial_condition.type}' not implemented")
    
    return grid, solver, state


def compute_diagnostics(state, grid, solver):
    """Compute diagnostic quantities."""
    # Handle different state types
    if hasattr(state, 'base_state'):
        # Extended state with scalars
        theta_hat = state.theta_hat
        theta = ifft2(theta_hat)
    else:
        # Standard state
        theta_hat = state["theta_hat"]
        theta = ifft2(theta_hat)
    
    # Compute basic diagnostics
    diagnostics = {
        "theta_rms": float(jnp.sqrt(jnp.mean(theta**2))),
        "theta_max": float(jnp.max(jnp.abs(theta))),
    }
    
    # Add energy and enstrophy
    diagnostics["energy"] = compute_total_energy(theta_hat, grid, solver.alpha)
    diagnostics["enstrophy"] = compute_enstrophy(theta_hat, grid, solver.alpha)
    
    # Add solver diagnostics if available
    if hasattr(solver, 'get_diagnostics'):
        solver_diags = solver.get_diagnostics(state)
        diagnostics.update(solver_diags)
    
    return diagnostics


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--device", type=click.Choice(["cpu", "gpu", "tpu"]), default="cpu",
              help="Device to run on")
@click.option("--checkpoint", type=click.Path(exists=True), 
              help="Path to checkpoint file to resume from")
@click.option("--output-dir", type=click.Path(), default="./output",
              help="Directory for output files")
@click.option("--dry-run", is_flag=True,
              help="Validate configuration without running")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Console logging level")
def main(config, device, checkpoint, output_dir, dry_run, log_level):
    """Run pygSQuiG simulation from YAML configuration file.
    
    Example:
        pygsquig-run config.yml --device=gpu
        pygsquig-run config.yml --checkpoint=output/checkpoints/step_1000.h5
    """
    # Ensure global variable is accessible
    global shutdown_requested
    
    # Load configuration first
    run_config = load_config(config)
    
    # Set up output directories and logging
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir if not dry_run else None, console_level=log_level)
    
    # Make logger available to signal handler
    signal_handler.logger = logger
    
    # Set up device
    if device == "gpu":
        if not jax.devices("gpu"):
            logger.warning("No GPU found, falling back to CPU")
            device = "cpu"
        else:
            logger.info(f"Using GPU: {jax.devices('gpu')[0]}")
    elif device == "tpu":
        if not jax.devices("tpu"):
            logger.warning("No TPU found, falling back to CPU")
            device = "cpu"
    
    logger.info(f"Loading configuration: {config}")
    
    if dry_run:
        logger.info("Dry run mode - validating configuration only")
        logger.info("Configuration validated successfully!")
        logger.info(f"Grid: {run_config.grid.N}x{run_config.grid.N}, L={run_config.grid.L}")
        logger.info(f"Solver: α={run_config.solver.alpha}")
        logger.info(f"Simulation time: {run_config.simulation.t_end}")
        return
    
    # Set up output directories
    dirs = setup_output_directories(output_dir, run_config, logger)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize simulation
    logger.info("Initializing simulation...")
    grid, solver, state = initialize_simulation(run_config, checkpoint, logger)
    
    # Log simulation start
    logger.log_simulation_start(run_config, device)
    
    # Get time accessor based on state type
    get_time = lambda s: s.time if hasattr(s, 'time') else s["time"]
    get_step = lambda s: s.step if hasattr(s, 'step') else s["step"]
    get_theta_hat = lambda s: s.theta_hat if hasattr(s, 'theta_hat') else s["theta_hat"]
    
    # Set up timing
    t_start = time.time()
    initial_time = get_time(state) if hasattr(state, 'time') or 'time' in state else 0.0
    t_last_output = initial_time
    t_last_checkpoint = initial_time
    t_last_log = initial_time
    
    # Initialize forcing if configured
    forcing_fn = None
    rng_key = None
    if run_config.forcing and run_config.forcing.type != "none":
        if run_config.forcing.type == "ring":
            forcing_fn = RingForcing(
                kf=run_config.forcing.kf,
                dk=run_config.forcing.dk,
                epsilon=run_config.forcing.epsilon,
                tau_f=run_config.forcing.tau_f
            )
            # Initialize RNG key for forcing
            rng_seed = run_config.forcing.seed if run_config.forcing.seed else 42
            rng_key = jax.random.PRNGKey(rng_seed)
            logger.info(f"Initialized ring forcing: kf={run_config.forcing.kf}, ε={run_config.forcing.epsilon}")
        else:
            logger.warning(f"Unknown forcing type: {run_config.forcing.type}")
    
    # Initialize damping if configured  
    damping_fn = None
    if run_config.solver.damping and run_config.solver.damping.type != "none":
        damping_fn = CombinedDamping(
            mu=run_config.solver.damping.mu,
            kf=run_config.forcing.kf if run_config.forcing else 20.0,
            nu_p=run_config.solver.dissipation.nu_p,
            p=run_config.solver.dissipation.p
        )
        logger.info(f"Initialized damping: μ={run_config.solver.damping.mu}")
    
    logger.info(f"Starting simulation from t={initial_time:.2f} to t={run_config.simulation.t_end:.2f}")
    logger.info("Press Ctrl+C to save and exit gracefully")
    
    while get_time(state) < run_config.simulation.t_end and not shutdown_requested:
        # Compute time step
        if run_config.solver.time_integration.adaptive_cfl:
            from pygsquig.core.operators import compute_velocity_from_theta
            u, v = compute_velocity_from_theta(get_theta_hat(state), grid, run_config.solver.alpha)
            dt = adaptive_timestep(
                u, v, grid.x[1, 0] - grid.x[0, 0], grid.y[0, 1] - grid.y[0, 0],
                run_config.solver.time_integration.cfl_safety,
                run_config.solver.time_integration.dt_max
            )
        else:
            dt = run_config.solver.time_integration.dt
        
        # Don't overshoot end time
        dt = min(dt, run_config.simulation.t_end - get_time(state))
        
        # Time step with forcing (need to pass RNG key)
        if forcing_fn and rng_key is not None:
            # Split key for this timestep
            rng_key, subkey = jax.random.split(rng_key)
            # Create forcing lambda that includes the key
            forcing_with_key = lambda theta_hat: forcing_fn(theta_hat, subkey, dt, grid)
            state = solver.step(state, dt, forcing=forcing_with_key, damping=damping_fn)
        else:
            state = solver.step(state, dt, forcing=forcing_fn, damping=damping_fn)
        
        # For extended states, also step velocity if needed
        if hasattr(state, 'base_state') and hasattr(solver, 'scalars'):
            # Compute velocity for scalar advection
            from pygsquig.core.operators import compute_velocity_from_theta
            u, v = compute_velocity_from_theta(state.theta_hat, grid, run_config.solver.alpha)
        
        # Logging
        current_time = get_time(state)
        if current_time - t_last_log >= run_config.simulation.log_interval:
            diagnostics = compute_diagnostics(state, grid, solver)
            elapsed = time.time() - t_start
            
            # Compute ETA
            if current_time > t_last_log:
                rate = (current_time - t_last_log) / (time.time() - t_start + elapsed)
                eta = (run_config.simulation.t_end - current_time) / rate if rate > 0 else 0
            else:
                eta = None
            
            logger.log_progress(
                current_time, 
                get_step(state), 
                dt, 
                diagnostics,
                eta
            )
            
            # Save diagnostics
            save_diagnostics(
                diagnostics,
                current_time,
                dirs["diagnostics"] / "timeseries.h5"
            )
            
            t_last_log = current_time
        
        # Output fields
        if get_time(state) - t_last_output >= run_config.simulation.output_interval:
            # Prepare fields to save
            data = {}
            
            # Handle different state types (with or without scalars)
            if hasattr(state, 'base_state'):
                # Extended state with scalars
                theta = ifft2(state.theta_hat)
            else:
                # Standard state
                theta = ifft2(state["theta_hat"])
            
            for field in run_config.output.fields:
                if field == "theta":
                    data["theta"] = theta
                elif field == "vorticity":
                    # TODO: Implement vorticity calculation
                    # For gSQG, vorticity = (-Δ)^(α/2) θ
                    logger.warning("Vorticity output not yet implemented")
                    continue
                elif field == "streamfunction":
                    from pygsquig.core.operators import compute_streamfunction
                    theta_hat = state.theta_hat if hasattr(state, 'theta_hat') else state["theta_hat"]
                    psi_hat = compute_streamfunction(theta_hat, grid, run_config.solver.alpha)
                    data["streamfunction"] = ifft2(psi_hat)
                elif field == "velocity":
                    from pygsquig.core.operators import compute_velocity_from_theta
                    theta_hat = state.theta_hat if hasattr(state, 'theta_hat') else state["theta_hat"]
                    u, v = compute_velocity_from_theta(theta_hat, grid, run_config.solver.alpha)
                    data["u"] = u
                    data["v"] = v
                elif field == "scalars" and hasattr(state, 'scalar_state') and state.scalar_state:
                    # Add all scalar fields
                    for name, scalar_hat in state.scalar_state.scalars.items():
                        data[f"scalar_{name}"] = ifft2(scalar_hat)
            
            # Save output
            step = state.step if hasattr(state, 'step') else state["step"]
            time_val = state.time if hasattr(state, 'time') else state["time"]
            output_file = dirs["fields"] / f"fields_{step:08d}.nc"
            metadata = {
                "step": int(step),  # Convert to Python int
                "alpha": float(run_config.solver.alpha),
                "nu_p": float(run_config.solver.dissipation.nu_p),
                "p": int(run_config.solver.dissipation.p),
            }
            save_output(data, grid, time_val, metadata, output_file, 
                       compress=run_config.output.compress)
            logger.log_output(output_file, time_val, step)
            
            t_last_output = get_time(state)
        
        # Checkpointing
        if get_time(state) - t_last_checkpoint >= run_config.simulation.checkpoint_interval:
            step = get_step(state)
            checkpoint_file = dirs["checkpoints"] / f"step_{step:08d}.h5"
            # Convert ExtendedState to dict if needed
            state_dict = state.to_dict() if hasattr(state, 'to_dict') else state
            save_checkpoint(state_dict, run_config, checkpoint_file)
            logger.log_checkpoint(checkpoint_file, get_time(state), step)
            t_last_checkpoint = get_time(state)
        
        # Check wall time limit
        if run_config.simulation.wall_time_limit:
            if time.time() - t_start > run_config.simulation.wall_time_limit:
                logger.warning("Wall time limit reached!")
                shutdown_requested = True
    
    # Final checkpoint and output
    logger.info("Saving final state...")
    
    # Save final checkpoint
    final_step = get_step(state)
    final_time = get_time(state)
    checkpoint_file = dirs["checkpoints"] / f"final_step_{final_step:08d}.h5"
    # Convert ExtendedState to dict if needed
    state_dict = state.to_dict() if hasattr(state, 'to_dict') else state
    save_checkpoint(state_dict, run_config, checkpoint_file)
    logger.info(f"Final checkpoint saved: {checkpoint_file}")
    
    # Save final output if needed
    if final_time - t_last_output > 0:
        theta_hat = get_theta_hat(state)
        data = {"theta": ifft2(theta_hat)}
        
        # Add scalars if present
        if hasattr(state, 'scalar_state') and state.scalar_state:
            for name, scalar_hat in state.scalar_state.scalars.items():
                data[f"scalar_{name}"] = ifft2(scalar_hat)
        
        output_file = dirs["fields"] / f"fields_final_{final_step:08d}.nc"
        save_output(data, grid, final_time, {"step": int(final_step)}, output_file)
        logger.info(f"Final output saved: {output_file}")
    
    # Summary
    elapsed = time.time() - t_start
    logger.log_simulation_complete(final_time, final_step, elapsed)
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
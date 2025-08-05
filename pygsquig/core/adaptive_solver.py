"""
Adaptive solver with CFL-based timestepping for pygSQuiG.

This module provides a solver wrapper that automatically adjusts
timesteps based on stability criteria.
"""

import time as pytime
from typing import Any, Callable, Optional

import jax

from pygsquig.core.adaptive_timestep import (
    AdaptiveTimestepper,
    CFLConfig,
    estimate_initial_timestep,
)
from pygsquig.core.grid import Grid
from pygsquig.core.operators import compute_velocity_from_theta
from pygsquig.core.solver import State, gSQGSolver


class AdaptivegSQGSolver:
    """gSQG solver with adaptive timestepping.

    This solver automatically adjusts timesteps to maintain stability
    while maximizing efficiency.
    """

    def __init__(
        self,
        grid: Grid,
        alpha: float,
        nu_p: float = 0.0,
        p: int = 8,
        cfl_config: Optional[CFLConfig] = None,
        verbose: bool = False,
    ):
        """Initialize adaptive solver.

        Args:
            grid: Grid object
            alpha: Fractional power
            nu_p: Hyperviscosity coefficient
            p: Hyperviscosity order
            cfl_config: CFL configuration
            verbose: Print timestep information
        """
        # Create base solver
        self.base_solver = gSQGSolver(grid, alpha, nu_p, p)

        # Store parameters
        self.grid = grid
        self.alpha = alpha
        self.nu_p = nu_p
        self.p = p
        self.verbose = verbose

        # Create adaptive timestepper
        self.timestepper = AdaptiveTimestepper(grid, cfl_config, verbose=verbose)

        # Current timestep
        self.current_dt = estimate_initial_timestep(
            grid, nu_p, p, target_cfl=cfl_config.target_cfl if cfl_config else 0.5
        )

        if verbose:
            print(f"Initial timestep estimate: {self.current_dt:.3e}")

    def initialize(self, **kwargs) -> State:
        """Initialize solver state.

        Args:
            **kwargs: Arguments passed to base solver

        Returns:
            Initial state
        """
        return self.base_solver.initialize(**kwargs)

    def step(
        self,
        state: State,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        **kwargs,
    ) -> tuple[State, dict[str, Any]]:
        """Advance solution with adaptive timestep.

        Args:
            state: Current state
            forcing: Optional forcing function
            damping: Optional damping function
            **kwargs: Additional arguments

        Returns:
            Tuple of (new_state, step_info)
        """
        # Compute velocity for CFL
        u, v = compute_velocity_from_theta(state["theta_hat"], self.grid, self.alpha)

        # Compute adaptive timestep
        dt, dt_diags = self.timestepper.compute_timestep(state, u, v, self.nu_p, self.p)

        # Store for potential retry
        state_backup = state.copy()

        # Attempt step with computed dt
        try:
            new_state = self.base_solver.step(
                state, dt, forcing=forcing, damping=damping, **kwargs
            )

            # Check stability
            is_stable, reason = self.timestepper.check_stability(state, new_state, dt)

            if not is_stable:
                # Retry with smaller timestep
                if self.verbose:
                    print(f"Step rejected: {reason}")

                dt_retry = self.timestepper.suggest_retry_timestep(dt)

                # Retry from backup
                new_state = self.base_solver.step(
                    state_backup, dt_retry, forcing=forcing, damping=damping, **kwargs
                )

                dt = dt_retry
                dt_diags["retry"] = True

        except Exception as e:
            # Handle numerical errors
            if self.verbose:
                print(f"Step failed with error: {e}")

            dt_retry = self.timestepper.suggest_retry_timestep(dt)

            # Retry with smaller timestep
            new_state = self.base_solver.step(
                state_backup, dt_retry, forcing=forcing, damping=damping, **kwargs
            )

            dt = dt_retry
            dt_diags["retry"] = True
            dt_diags["error"] = str(e)

        # Update current timestep
        self.current_dt = dt

        # Prepare step info
        step_info = {
            "dt_used": dt,
            "dt_diagnostics": dt_diags,
            "time": new_state["time"],
            "step": new_state["step"],
        }

        return new_state, step_info

    def evolve(
        self,
        state: State,
        t_final: float,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        save_interval: Optional[float] = None,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Evolve system to final time with adaptive stepping.

        Args:
            state: Initial state
            t_final: Target final time
            forcing: Optional forcing function
            damping: Optional damping function
            save_interval: Time interval for saving states
            callback: Optional callback function(state, info)
            **kwargs: Additional arguments

        Returns:
            Dictionary with results and statistics
        """
        # Initialize storage
        results: dict[str, Any] = {"times": [], "states": [], "dt_history": [], "diagnostics": []}

        # Timing
        wall_time_start = pytime.perf_counter()

        # Save initial state
        if save_interval is not None:
            results["times"].append(state["time"])
            results["states"].append(state.copy())
            next_save_time = state["time"] + save_interval
        else:
            next_save_time = t_final + 1  # Never save

        # Main evolution loop
        current_state = state
        step_count = 0

        if self.verbose:
            print(f"Evolving from t={state['time']} to t={t_final}")

        while current_state["time"] < t_final:
            # Compute timestep (may be limited by t_final)
            u, v = compute_velocity_from_theta(current_state["theta_hat"], self.grid, self.alpha)

            dt_adaptive, _ = self.timestepper.compute_timestep(
                current_state, u, v, self.nu_p, self.p
            )

            # Limit by remaining time
            dt = min(dt_adaptive, t_final - current_state["time"])

            # Also limit to not overshoot save time
            if save_interval is not None:
                dt = min(dt, next_save_time - current_state["time"])

            # Step
            new_state, step_info = self.step(
                current_state, forcing=forcing, damping=damping, **kwargs
            )

            # Update state
            current_state = new_state
            step_count += 1

            # Save if needed
            if save_interval is not None and current_state["time"] >= next_save_time:
                results["times"].append(current_state["time"])
                results["states"].append(current_state.copy())
                results["dt_history"].append(step_info["dt_used"])
                results["diagnostics"].append(step_info["dt_diagnostics"])
                next_save_time += save_interval

            # Callback
            if callback is not None:
                callback(current_state, step_info)

            # Progress reporting
            if self.verbose and step_count % 100 == 0:
                progress = (current_state["time"] - state["time"]) / (t_final - state["time"])
                elapsed = pytime.perf_counter() - wall_time_start
                eta = elapsed / progress - elapsed if progress > 0 else 0

                print(
                    f"  Step {step_count}: t={current_state['time']:.3f} "
                    f"({progress*100:.1f}%), dt={step_info['dt_used']:.3e}, "
                    f"CFL={step_info['dt_diagnostics']['cfl_adv']:.3f}, "
                    f"ETA={eta:.1f}s"
                )

        # Save final state if not already saved
        if save_interval is not None and current_state["time"] > results["times"][-1]:
            results["times"].append(current_state["time"])
            results["states"].append(current_state.copy())

        # Add statistics
        wall_time_total = pytime.perf_counter() - wall_time_start
        stats = self.timestepper.get_statistics()
        stats["total_steps"] = step_count
        stats["wall_time"] = wall_time_total
        stats["simulated_time"] = current_state["time"] - state["time"]
        stats["time_ratio"] = stats["simulated_time"] / wall_time_total

        results["statistics"] = stats
        results["final_state"] = current_state

        if self.verbose:
            print("\nEvolution complete:")
            print(f"  Total steps: {step_count}")
            print(f"  Wall time: {wall_time_total:.2f}s")
            print(f"  Mean timestep: {stats['dt_mean']:.3e}")
            print(f"  Efficiency: {stats['efficiency']*100:.1f}%")

        return results

    def get_diagnostics(self, state: State) -> dict[str, float]:
        """Get solver diagnostics including CFL information.

        Args:
            state: Current state

        Returns:
            Dictionary of diagnostics
        """
        # Base diagnostics
        diags = self.base_solver.get_diagnostics(state)

        # Add CFL diagnostics
        u, v = compute_velocity_from_theta(state["theta_hat"], self.grid, self.alpha)

        _, dt_diags = self.timestepper.compute_timestep(state, u, v, self.nu_p, self.p)

        diags.update(
            {f"cfl_{key}": value for key, value in dt_diags.items() if key.startswith("cfl_")}
        )

        diags["current_dt"] = self.current_dt
        diags["max_velocity"] = dt_diags["max_velocity"]

        return diags


def adaptive_solver_example():
    """Example usage of adaptive solver."""
    import numpy as np

    from pygsquig.core.grid import make_grid
    from pygsquig.forcing.ring_forcing import RingForcing

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Configure CFL
    cfl_config = CFLConfig(
        cfl_safety=0.8, target_cfl=0.5, dt_min=1e-6, dt_max=0.01, growth_factor=1.2
    )

    # Create adaptive solver
    solver = AdaptivegSQGSolver(
        grid, alpha=1.0, nu_p=1e-6, p=8, cfl_config=cfl_config, verbose=True
    )

    # Initialize
    state = solver.initialize(seed=42)

    # Add forcing
    forcing = RingForcing(kf=20.0, dk=2.0, epsilon=0.1)

    # Evolve with adaptive stepping
    results = solver.evolve(
        state,
        t_final=10.0,
        forcing=forcing,
        save_interval=1.0,
        key=jax.random.PRNGKey(123),  # For forcing
    )

    # Print statistics
    print("\nAdaptive timestepping statistics:")
    for key, value in results["statistics"].items():
        print(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    # Run example
    results = adaptive_solver_example()

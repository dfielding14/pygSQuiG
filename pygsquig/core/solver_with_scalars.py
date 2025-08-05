"""
Extended gSQG solver with passive scalar support.

This module extends the base gSQG solver to include passive scalar
evolution capabilities while maintaining backward compatibility.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp

from pygsquig.core.solver import State, gSQGSolver
from pygsquig.exceptions import ConfigurationError
from pygsquig.scalars.passive_scalar import MultiSpeciesEvolver, PassiveScalarEvolver
from pygsquig.scalars.state import MultiScalarState


@dataclass
class ExtendedState:
    """Extended simulation state including passive scalars.

    This wraps the base state and adds scalar fields while
    maintaining compatibility with the base solver.

    Attributes:
        base_state: Core gSQG state (theta_hat, time, step)
        scalar_state: Optional passive scalar state
    """

    base_state: State
    scalar_state: Optional[MultiScalarState] = None

    @property
    def theta_hat(self) -> jnp.ndarray:
        """Access active scalar for compatibility."""
        return self.base_state["theta_hat"]

    @property
    def time(self) -> float:
        """Access time for compatibility."""
        return float(self.base_state["time"])

    @property
    def step(self) -> int:
        """Access step count for compatibility."""
        return int(self.base_state["step"])

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = dict(self.base_state)
        if self.scalar_state is not None:
            result["scalars"] = self.scalar_state.scalars
        return result


class gSQGSolverWithScalars(gSQGSolver):
    """gSQG solver extended with passive scalar capabilities.

    This solver maintains full compatibility with the base gSQGSolver
    while adding optional passive scalar evolution.

    Additional Attributes:
        scalar_evolver: Optional MultiSpeciesEvolver for passive scalars
    """

    scalar_evolver: Optional[MultiSpeciesEvolver]

    def __init__(
        self,
        grid,
        alpha: float,
        nu_p: float = 0.0,
        p: int = 8,
        passive_scalars: Optional[dict[str, dict]] = None,
    ):
        """Initialize solver with optional passive scalars.

        Args:
            grid: Grid object
            alpha: Fractional power for gSQG dynamics
            nu_p: Hyperviscosity coefficient
            p: Hyperviscosity order
            passive_scalars: Dictionary of scalar configurations
                Format: {name: {'kappa': float, 'source': SourceTerm}}
        """
        # Initialize base solver
        super().__init__(grid, alpha, nu_p, p)

        # Setup passive scalars if provided
        if passive_scalars is not None:
            self.scalar_evolver = MultiSpeciesEvolver(grid=grid, species=passive_scalars)
        else:
            self.scalar_evolver = None

    def initialize(
        self,
        theta0: Optional[jnp.ndarray] = None,
        seed: Optional[int] = None,
        scalar_init: Optional[dict[str, jnp.ndarray]] = None,
    ) -> dict[str, Any]:
        """Initialize simulation state with optional scalars.

        Args:
            theta0: Initial condition for active scalar
            seed: Random seed
            scalar_init: Initial conditions for passive scalars
                Format: {name: field_array}

        Returns:
            State dictionary, potentially with scalar fields added
        """
        # Initialize base state
        base_state = super().initialize(theta0, seed)

        # Return base state if no scalars
        if self.scalar_evolver is None:
            return base_state

        # Initialize scalar state
        if scalar_init is None:
            scalar_init = {}

        scalar_state = self.scalar_evolver.initialize(scalar_init)

        # Return state with scalar fields added
        extended_state = dict(base_state)
        extended_state["scalar_state"] = scalar_state
        return extended_state

    def step(
        self,
        state: dict[str, Any],
        dt: float,
        forcing: Optional[Callable] = None,
        damping: Optional[Callable] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Advance solution by one time step.

        Args:
            state: Current state (base or extended)
            dt: Time step size
            forcing: Optional forcing for active scalar
            damping: Optional damping for active scalar
            **kwargs: Additional arguments for forcing/damping

        Returns:
            Updated state dictionary
        """
        # Check if state has scalar fields
        if "scalar_state" not in state or state["scalar_state"] is None:
            # No scalars - use base solver
            return super().step(state, dt, forcing, damping, **kwargs)

        # Extract scalar state
        scalar_state = state["scalar_state"]

        # Create base state dict without scalar fields
        base_state = {k: v for k, v in state.items() if k != "scalar_state"}

        # Step active scalar
        new_base_state = super().step(base_state, dt, forcing, damping, **kwargs)

        # Step passive scalars if present
        if self.scalar_evolver is not None and scalar_state is not None:
            # Compute velocity at new time
            u, v = self.compute_velocity(new_base_state["theta_hat"])

            # Step all scalars
            new_scalar_state = self.scalar_evolver.step(scalar_state, dt, u, v)
        else:
            new_scalar_state = scalar_state

        # Return updated state with scalar fields
        result = dict(new_base_state)
        result["scalar_state"] = new_scalar_state
        return result

    def get_diagnostics(self, state: dict[str, Any]) -> dict[str, float]:
        """Compute diagnostics including passive scalars.

        Args:
            state: Current state

        Returns:
            Dictionary of all diagnostics
        """
        # Get base diagnostics
        # Create base state without scalar fields for diagnostics
        base_state = {k: v for k, v in state.items() if k != "scalar_state"}
        base_diagnostics = super().get_diagnostics(base_state)

        # Add scalar diagnostics if present
        if (
            "scalar_state" in state
            and state["scalar_state"] is not None
            and self.scalar_evolver is not None
        ):
            scalar_state = state["scalar_state"]
            # Get diagnostics for each scalar
            for name, scalar_hat in scalar_state.scalars.items():
                evolver = self.scalar_evolver.evolvers[name]
                from pygsquig.scalars.state import PassiveScalarState

                ps_state = PassiveScalarState(scalar_hat=scalar_hat, time=state["time"])
                scalar_diags = evolver.get_diagnostics(ps_state)
                base_diagnostics.update(scalar_diags)

        return base_diagnostics

    def add_passive_scalar(
        self, name: str, kappa: float = 0.0, source: Optional[Any] = None
    ) -> None:
        """Add a new passive scalar to existing solver.

        Args:
            name: Name identifier for the scalar
            kappa: Diffusivity coefficient
            source: Optional source term

        Raises:
            ConfigurationError: If scalar name already exists
        """
        if self.scalar_evolver is None:
            # Create new evolver
            self.scalar_evolver = MultiSpeciesEvolver(
                grid=self.grid, species={name: {"kappa": kappa, "source": source}}
            )
        else:
            # Add to existing evolver
            if name in self.scalar_evolver.evolvers:
                raise ConfigurationError(f"Scalar '{name}' already exists")
            self.scalar_evolver.evolvers[name] = PassiveScalarEvolver(
                grid=self.grid, kappa=kappa, source_fn=source, name=name
            )

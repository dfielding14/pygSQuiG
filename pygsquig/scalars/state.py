"""
State management for passive scalars.

This module defines immutable state containers optimized for JAX.
"""

from typing import NamedTuple, Dict, Optional
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node


class PassiveScalarState(NamedTuple):
    """Immutable state for a single passive scalar.

    Attributes:
        scalar_hat: Fourier coefficients of the scalar field
        time: Current simulation time
        total_source: Integrated source term (for conservation checks)
        total_dissipation: Integrated dissipation (for conservation checks)
    """

    scalar_hat: jnp.ndarray
    time: float
    total_source: float = 0.0
    total_dissipation: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary representation of state
        """
        return {
            "scalar_hat": self.scalar_hat,
            "time": self.time,
            "total_source": self.total_source,
            "total_dissipation": self.total_dissipation,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict) -> "PassiveScalarState":
        """Create from dictionary.

        Args:
            state_dict: Dictionary with state components

        Returns:
            PassiveScalarState instance
        """
        return cls(
            scalar_hat=state_dict["scalar_hat"],
            time=state_dict.get("time", 0.0),
            total_source=state_dict.get("total_source", 0.0),
            total_dissipation=state_dict.get("total_dissipation", 0.0),
        )


class MultiScalarState(NamedTuple):
    """Immutable state for multiple passive scalars.

    Attributes:
        scalars: Dictionary mapping scalar names to their Fourier coefficients
        time: Current simulation time
        diagnostics: Dictionary of accumulated diagnostics per scalar
    """

    scalars: Dict[str, jnp.ndarray]
    time: float
    diagnostics: Optional[Dict[str, Dict[str, float]]] = None

    def get_scalar(self, name: str) -> jnp.ndarray:
        """Get a specific scalar field.

        Args:
            name: Name of the scalar

        Returns:
            Fourier coefficients of the scalar

        Raises:
            KeyError: If scalar name not found
        """
        if name not in self.scalars:
            raise KeyError(f"Scalar '{name}' not found in state")
        return self.scalars[name]

    def update_scalar(self, name: str, scalar_hat: jnp.ndarray) -> "MultiScalarState":
        """Create new state with updated scalar.

        Args:
            name: Name of scalar to update
            scalar_hat: New Fourier coefficients

        Returns:
            New MultiScalarState with updated scalar
        """
        new_scalars = dict(self.scalars)
        new_scalars[name] = scalar_hat
        return self._replace(scalars=new_scalars)


# Register PassiveScalarState as JAX pytree
def _scalar_state_flatten(state: PassiveScalarState):
    """Flatten PassiveScalarState for JAX."""
    children = [state.scalar_hat]
    aux_data = (state.time, state.total_source, state.total_dissipation)
    return children, aux_data


def _scalar_state_unflatten(aux_data, children):
    """Unflatten PassiveScalarState for JAX."""
    scalar_hat = children[0]
    time, total_source, total_dissipation = aux_data
    return PassiveScalarState(scalar_hat, time, total_source, total_dissipation)


register_pytree_node(PassiveScalarState, _scalar_state_flatten, _scalar_state_unflatten)


# Register MultiScalarState as JAX pytree
def _multi_state_flatten(state: MultiScalarState):
    """Flatten MultiScalarState for JAX."""
    # Extract scalar arrays in consistent order
    names = sorted(state.scalars.keys())
    children = [state.scalars[name] for name in names]
    aux_data = (names, state.time, state.diagnostics)
    return children, aux_data


def _multi_state_unflatten(aux_data, children):
    """Unflatten MultiScalarState for JAX."""
    names, time, diagnostics = aux_data
    scalars = {name: child for name, child in zip(names, children)}
    return MultiScalarState(scalars, time, diagnostics)


register_pytree_node(MultiScalarState, _multi_state_flatten, _multi_state_unflatten)

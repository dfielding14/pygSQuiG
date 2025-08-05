"""
pygSQuiG: A Python/JAX solver for generalised Surface-Quasi-Geostrophic turbulence.
"""

__version__ = "0.1.0"
__author__ = "pygSQuiG Developers"

# Version check for JAX
import warnings

import jax

if not hasattr(jax, "__version__") or jax.__version__ < "0.4.0":
    warnings.warn(
        "pygSQuiG requires JAX >= 0.4.0. " "Please upgrade with: pip install --upgrade jax",
        RuntimeWarning,
        stacklevel=2,
    )

# Enable double precision by default
try:
    from jax import config

    config.update("jax_enable_x64", True)
except ImportError:
    # Older JAX versions
    import jax.config

    jax.config.update("jax_enable_x64", True)

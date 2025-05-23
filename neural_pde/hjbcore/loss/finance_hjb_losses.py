"""Loss functions specific to financial HJB equations.

These are placeholders and will be replaced with full implementations.
"""

from typing import Any, Dict
import jax.numpy as jnp


def finance_hjb_loss(model: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return zero as placeholder for finance HJB loss."""
    return jnp.array(0.0)

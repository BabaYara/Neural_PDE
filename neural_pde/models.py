"""Minimal models for Neural PDE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class FBSDE:
    """Simple FBSDE model placeholder."""

    params: jnp.ndarray

    def __init__(self, key: jax.random.KeyArray):
        self.params = jnp.ones(1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x) * self.params[0]

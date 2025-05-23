"""Simple Euler integrator for SDEs."""
from typing import Callable
import jax.numpy as jnp


def euler_step(x, t, dt, drift: Callable, diffusion: Callable, key=None):
    """Perform one Euler-Maruyama step."""
    dw = jnp.sqrt(dt) * jnp.asarray(0.0 if key is None else key)
    return x + drift(x, t) * dt + diffusion(x, t) * dw

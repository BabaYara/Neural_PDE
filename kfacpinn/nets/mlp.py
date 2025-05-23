"""Simple MLP network."""
from typing import Sequence
import jax
import jax.numpy as jnp


def init_mlp(sizes: Sequence[int], key) -> list:
    params = []
    keys = jax.random.split(key, len(sizes)-1)
    for in_dim, out_dim, k in zip(sizes[:-1], sizes[1:], keys):
        w_key, b_key = jax.random.split(k)
        W = jax.random.normal(w_key, (in_dim, out_dim)) / jnp.sqrt(in_dim)
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, W) + b)
    W, b = params[-1]
    return jnp.dot(x, W) + b

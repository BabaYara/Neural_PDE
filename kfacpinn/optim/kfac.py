from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp


@dataclass
class LayerState:
    cov_A: jnp.ndarray
    cov_G: jnp.ndarray


@dataclass
class KFACState:
    layer_states: List[LayerState]


class KFAC:
    """Very small KFAC-like optimizer."""

    def __init__(self, lr: float = 1e-3, momentum: float = 0.95, damping: float = 1e-3):
        self.lr = lr
        self.momentum = momentum
        self.damping = damping

    def init(self, params: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> KFACState:
        layer_states = []
        for W, _ in params:
            cov_A = jnp.eye(W.shape[0])
            cov_G = jnp.eye(W.shape[1])
            layer_states.append(LayerState(cov_A, cov_G))
        return KFACState(layer_states)

    def update(
        self,
        params: List[Tuple[jnp.ndarray, jnp.ndarray]],
        state: KFACState,
        grads: List[Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> Tuple[List[Tuple[jnp.ndarray, jnp.ndarray]], KFACState]:
        new_params = []
        new_states = []
        for (W, b), (gW, gb), s in zip(params, grads, state.layer_states):
            cov_A = (1 - self.momentum) * jnp.eye(s.cov_A.shape[0]) + self.momentum * s.cov_A
            cov_G = (1 - self.momentum) * jnp.eye(s.cov_G.shape[0]) + self.momentum * s.cov_G

            pre_gW = jnp.linalg.solve(cov_A + self.damping * jnp.eye(cov_A.shape[0]), gW)
            pre_gW = jnp.linalg.solve((cov_G + self.damping * jnp.eye(cov_G.shape[0])).T, pre_gW.T).T

            new_W = W - self.lr * pre_gW
            new_b = b - self.lr * gb
            new_params.append((new_W, new_b))
            new_states.append(LayerState(cov_A, cov_G))
        return new_params, KFACState(new_states)

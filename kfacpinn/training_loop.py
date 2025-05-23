"""Simple training loop skeleton for PINNs."""
from typing import Any, Sequence

import optax
import jax

from .loss.builder import make_loss


class Trainer:
    def __init__(self, model, regs: Sequence[str]):
        self.model = model
        self.loss_fn = make_loss(regs)
        self.opt = optax.adam(1e-3)

    def fit(self, params, state, num_steps: int = 1000):
        opt_state = self.opt.init(params)

        @jax.jit
        def step(params, opt_state, state):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, state)
            updates, opt_state_ = self.opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state_, loss

        for _ in range(num_steps):
            params, opt_state, loss = step(params, opt_state, state)
        return params

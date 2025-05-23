"""Simple training loop skeleton for PINNs."""

from typing import Any, Dict

import optax
import jax

from neural_pde.hjbcore.loss_suite import LossSuite
from .optim.kfac import KFAC, KFACState


class Trainer:
    def __init__(self, model: Any, cfg: Dict[str, Any], use_kfac: bool = False):
        self.model = model
        self.loss_suite = LossSuite(cfg)
        if use_kfac:
            self.opt = KFAC(lr=cfg.get("lr", 1e-3))
        else:
            self.opt = optax.adam(cfg.get("lr", 1e-3))

    def fit(self, batch: Dict[str, Any], num_steps: int = 1000):
        params = self.model.params

        if isinstance(self.opt, KFAC):
            opt_state = self.opt.init(params)

            def step(params, opt_state, batch):
                self.model.params = params
                loss, grads = jax.value_and_grad(
                    lambda p: self.loss_suite.total_loss(self.model, batch)
                )(params)
                params, opt_state = self.opt.update(params, opt_state, grads)
                return params, opt_state, loss
        else:
            opt_state = self.opt.init(params)

            @jax.jit
            def step(params, opt_state, batch):
                self.model.params = params
                loss, grads = jax.value_and_grad(
                    lambda p: self.loss_suite.total_loss(self.model, batch)
                )(params)
                updates, opt_state_ = self.opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state_, loss

        for epoch in range(num_steps):
            params, opt_state, loss = step(params, opt_state, batch)
            self.loss_suite.iter_ctr = epoch

        self.model.params = params
        return params

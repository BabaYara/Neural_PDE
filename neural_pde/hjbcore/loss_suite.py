"""Composable loss suite for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp


@dataclass
class LossSuite:
    """Aggregates regularization losses."""

    cfg: Dict[str, Any]
    iter_ctr: int = 0

    def total_loss(self, model: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute total loss from all regularizers."""
        from .regularizers.mallavin import malliavin_loss
        from .regularizers.stein import stein_loss
        from .regularizers.signature import signature_loss
        from .regularizers.poisson_cv import poisson_cv_loss
        from .regularizers.sobolev import sobolev_loss

        base_loss = jnp.sum(model(batch["state"])) * self.cfg.get("dt", 1.0)
        loss = base_loss

        loss += self.cfg.get("λσ2", 0.0) * malliavin_loss(model, batch)
        loss += self.cfg.get("λμ", 0.0) * stein_loss(model, batch)

        sig_warmup = self.cfg.get("signature_warmup_iters", 0)
        sig_factor = 1.0
        if sig_warmup > 0:
            sig_factor = jnp.clip(self.iter_ctr / sig_warmup, 0.0, 1.0)
        loss += sig_factor * self.cfg.get("λsig", 0.0) * signature_loss(model, batch)

        loss += self.cfg.get("λpcv", 0.0) * poisson_cv_loss(model, batch)
        loss += self.cfg.get("λsob", 0.0) * sobolev_loss(model, batch)
        return loss

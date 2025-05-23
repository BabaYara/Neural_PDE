"""Loss builder for PINNs with regularizers."""
from typing import Callable, Sequence

from . import *  # noqa
from ..regularizers.mallavin import malliavin_loss
from ..regularizers.stein import stein_loss
from ..regularizers.signature import signature_loss
from ..regularizers.sobolev import sobolev_loss
from ..regularizers.poisson_cv import poisson_cv_loss

_REGISTRY = {
    "mallavin": malliavin_loss,
    "stein": stein_loss,
    "signature": signature_loss,
    "sobolev": sobolev_loss,
    "poisson_cv": poisson_cv_loss,
}


def make_loss(regs: Sequence[str]) -> Callable:
    funcs = [_REGISTRY[name] for name in regs]

    def loss_fn(params, state):
        total = 0.0
        for f in funcs:
            total += f(params, state)
        return total

    return loss_fn

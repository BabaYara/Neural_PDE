"""Top-level package for kfacpinn."""
from .nets.mlp import init_mlp, mlp
from .training_loop import Trainer

__all__ = ["init_mlp", "mlp", "Trainer"]

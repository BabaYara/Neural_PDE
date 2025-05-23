"""Loss functions for HJB-style PINN models."""

from .finance_hjb_losses import finance_hjb_loss

__all__ = ["finance_hjb_loss"]

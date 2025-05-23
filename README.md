# Neural PDE

A package for experimenting with different solution techniques for SDE using physics-informed neural networks techniques.

This repository now includes a lightweight implementation of the KFAC optimizer
tailored for PINNs. The optimizer can be accessed via `kfacpinn.optim.KFAC` and
plugged into the training loop.

## Installation

```bash
pip install -e .
```

## Quick Start

See the notebooks in `examples/` for how to build and train a simple model.
`examples/01_single_tree.ipynb` introduces the modular loss suite, while
`examples/03_finance_hjb.ipynb` demonstrates the finance-specific HJB loss.
The new `examples/04_kfac_optimizer.ipynb` shows how to train with the built-in
KFAC optimiser.

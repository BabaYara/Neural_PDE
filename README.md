# KFAC PINN

A package for experimenting with physics-informed neural networks using a plug-and-play regularizer system. The project is inspired by a composable HJB solver layout.

## Installation

```bash
pip install -e .
```

## Quick Start

See the notebooks in `examples/` for how to build and train a simple model.

Modular loss suite supports Malliavin, Stein, Signature regularisation; see `examples/01_single_tree.ipynb`.

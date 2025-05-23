# Project Progress Tracker

This file documents the ongoing tasks for the KFAC-based PINN package.

## TODO
- [ ] Phase 0: create empty branch `modular-loss-suite` and add folder skeleton
- [x] Phase 0: add `finance_hjb_losses.py` to `hjbcore/loss/`
- [ ] Phase 1: refactor monolithic code into modules
- [ ] Phase 1: wire `LossSuite` into training loop
- [ ] Phase 1: add `tests/test_loss_smoke.py`
- [ ] Phase 2: create `configs/base.yaml` and load with `yaml.safe_load`
- [ ] Phase 3: implement regulariser stubs in `regularizers/`
- [ ] Phase 4: benchmarking notebook in `examples/01_single_tree.ipynb`

## Completed
- [x] Initial scaffold for KFAC PINN package (commit 26188ff)
- [x] Added modular loss suite and initial regularizer placeholders (commit 71b00e6)


"""Extreme Hierarchical Weight Sparsity -- Unstructured (EHWS-U).

A two-phase ADMM pruning method for causal LLMs, same as the
semi-structured variant, but Phase 2 spends one *global* sparsity budget
across the whole network instead of forcing every layer to the same
ratio:

- Phase 1 -- layer-wise ADMM: each prunable nn.Linear is processed
  sequentially, one continuous run straight at a fixed per-layer target
  sparsity, with the ADMM penalty cosine-ramped from 0 up to its max
  value over the run. Independent of whatever overall sparsity Phase 2
  will eventually target, so it only needs to run once per model.
- Phase 2 -- global ADMM: every layer's dense proxy is jointly
  re-optimized under one global loss, warm-started from Phase 1, with a
  single global weight-count budget shared across every layer
  (`global_projection.py`). Layers that turn out to be more redundant
  give up proportionally more of that budget; nothing forces any two
  layers to end up at the same sparsity, which is what makes the result
  genuinely unstructured.

See README.md for the full method write-up, hyperparameters, and
reproduction commands.
"""

from .diagonal_projection import diagonal_project
from .global_projection import global_diagonal_project
from .hessian import LayerHessian
from .model_layers import discover_prunable_layers, PrunableLayer
from .losses import combined_loss
from .admm import Phase1Config, Phase2Config, run_phase1, run_phase2

__all__ = [
    "LayerHessian",
    "diagonal_project",
    "global_diagonal_project",
    "discover_prunable_layers",
    "PrunableLayer",
    "combined_loss",
    "Phase1Config",
    "Phase2Config",
    "run_phase1",
    "run_phase2",
]

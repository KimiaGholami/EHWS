"""Extreme Hierarchical Weight Sparsity (EHWS) -- the semi-structured variant.

A two-phase ADMM pruning method for causal LLMs:

- Phase 1 -- layer-wise ADMM: each prunable nn.Linear is processed
  sequentially. Its dense proxy is trained against the true CE+KD loss
  rather than a per-layer reconstruction surrogate, and its sparse
  projection is a diagonal-Hessian-weighted magnitude score
  (`ehws/diagonal_projection.py`). A per-layer saturation rule lets each
  layer stop tightening its sparsity budget once it starts disagreeing
  with the true loss.
- Phase 2 -- global ADMM: every layer's dense proxy is then jointly
  re-optimized under one global loss (a single forward pass through the
  whole network), warm-started from Phase 1, with one sparsity ratio
  shared uniformly by every layer -- "semi-structured" in the sense that
  every layer keeps the same fraction of weights, even though which
  specific weights survive within a layer is unstructured.

See README.md for the full method write-up, hyperparameters, and
reproduction commands.
"""

from .diagonal_projection import diagonal_project
from .hessian import LayerHessian
from .model_layers import discover_prunable_layers, PrunableLayer
from .losses import combined_loss
from .admm import Phase1Config, Phase2Config, run_phase1, run_phase2

__all__ = [
    "LayerHessian",
    "diagonal_project",
    "discover_prunable_layers",
    "PrunableLayer",
    "combined_loss",
    "Phase1Config",
    "Phase2Config",
    "run_phase1",
    "run_phase2",
]

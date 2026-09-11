"""EHWS -- semi-structured (uniform-ratio) ADMM pruning.

Two-phase ADMM pruning method for causal language models that trains
directly against the true next-token loss under a hard sparsity
constraint. Extends ELSA (*The Unseen Frontier: Pushing the Limits of
LLM Sparsity with Surrogate-Free ADMM*) with a per-layer warm-start
(Phase 1) before a joint global fine-tune (Phase 2), both sharing one
uniform sparsity ratio across every layer. See README.md for the full
method write-up, and PROGRESS.md (in the sibling
`Extreme_Layer_Global_Pruning` package this was synced from) for the
session-by-session history behind the current design.
"""

from .admm import LayerState, Phase1Config, Phase2Config, build_dense_states, run_phase1, run_phase2
from .diagonal_projection import diagonal_project
from .hessian import LayerHessian
from .losses import combined_loss
from .model_layers import PrunableLayer, disable_fused_kernels, discover_prunable_layers
from .obs_projection import obs_project

__all__ = [
    "LayerHessian",
    "diagonal_project",
    "obs_project",
    "discover_prunable_layers",
    "disable_fused_kernels",
    "PrunableLayer",
    "combined_loss",
    "LayerState",
    "Phase1Config",
    "Phase2Config",
    "build_dense_states",
    "run_phase1",
    "run_phase2",
]

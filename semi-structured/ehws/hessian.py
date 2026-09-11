"""Calibration-data Hessian accumulation for the Z-step.

Both the CWS background (slide 17: ``H = X Xᵀ``) and the Proposed Work's
Phase 1 Z-step (slide 39, Eq. 2: ``argmin (z_l - v_l)ᵀ H_l (z_l - v_l)``)
use the standard SparseGPT/OBS reconstruction Hessian: the second-moment
matrix of a layer's input activations over calibration data,

    H_l = (2 / N) * sum_n x_n x_n^T

accumulated with a forward pre-hook on the layer while calibration batches
are fed through the (partially-pruned) network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerHessian:
    """Accumulates H = (2/N) X^T X for one nn.Linear's input activations."""

    def __init__(self, in_features: int, device: torch.device):
        self.device = device
        self.d_in = in_features
        self.H = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)
        self.n_samples = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Fold in one batch of input activations, shape (..., d_in)."""
        x = x.reshape(-1, x.shape[-1]).to(device=self.device, dtype=torch.float64)
        n = x.shape[0]
        if n == 0:
            return
        # Running mean-of-outer-products update so batches of different
        # sizes combine correctly (matches SparseGPT's streaming formula).
        self.H.mul_(self.n_samples / (self.n_samples + n))
        self.n_samples += n
        x = x * (2.0 / self.n_samples) ** 0.5
        self.H.add_(x.t() @ x)

    def hook(self, module: nn.Module, inputs) -> None:
        self.update(inputs[0])

    def damped(self, damping: float = 0.01, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """H + damping * mean(diag(H)) * I, guarding against dead/zero-variance input channels.

        Returned in `dtype` (float32 by default), not the float64 the
        running accumulation itself uses -- accumulating thousands of
        outer products in float64 is what avoids catastrophic-cancellation
        error building up over the sum, but the *returned* matrix only
        ever gets read afterward (ranking scores in `diagonal_project`,
        upcast back to float32 there regardless -- or a Cholesky solve in
        `obs_project`), so a smaller storage dtype here costs no
        meaningful precision. This matters at real scale: a model with
        many/wide prunable layers (e.g. HGRN-1.3B's 168 layers, some
        d_in=5632) can need >10GB just to keep every finished layer's full
        (d_in x d_in) Hessian resident in float64 -- `admm.py`'s
        `_compute_hessian` passes the model's own load dtype here
        (`--dtype` in run_pipeline.py) so H's storage footprint tracks
        whatever the rest of the pipeline is using.
        """
        H = self.H.clone()
        idx = torch.arange(H.shape[0], device=H.device)
        dead = H[idx, idx] == 0
        H[dead, dead] = 1.0
        H[idx, idx] += damping * H[idx, idx].mean()
        return H.to(dtype)

    def inverse(self, damping: float = 0.01) -> torch.Tensor:
        """The *full* H^-1 needed by the CWS-style per-row greedy projection
        (slide 30: "one upfront H^-1, then O(1) work per weight").
        """
        H = self.damped(damping)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        # Symmetrize away float rounding asymmetry before it compounds
        # through thousands of greedy-elimination steps.
        return 0.5 * (Hinv + Hinv.t())

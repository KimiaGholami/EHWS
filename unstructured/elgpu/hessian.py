"""Calibration-data Hessian for the Z-step.

Each prunable layer's Z-step needs an estimate of how much the model's
output changes when a given input feature is zeroed out. We use the
same reconstruction Hessian SparseGPT and CWS use: the second-moment
matrix of the layer's input activations, accumulated over calibration
data,

    H = (2 / N) * sum_n x_n x_n^T

We accumulate it with a forward pre-hook on the layer while calibration
batches are run through the (possibly already partially pruned) network,
so it reflects the layer's actual input distribution at prune time.
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
        # Running mean-of-outer-products so batches of different sizes
        # combine correctly, regardless of how many samples came before.
        self.H.mul_(self.n_samples / (self.n_samples + n))
        self.n_samples += n
        x = x * (2.0 / self.n_samples) ** 0.5
        self.H.add_(x.t() @ x)

    def hook(self, module: nn.Module, inputs) -> None:
        self.update(inputs[0])

    def damped(self, damping: float = 0.01) -> torch.Tensor:
        """H + damping * mean(diag(H)) * I.

        Guards against input channels that never fired during
        calibration (diagonal entry of exactly 0), which would otherwise
        make that channel look infinitely important to keep.
        """
        H = self.H.clone()
        idx = torch.arange(H.shape[0], device=H.device)
        dead = H[idx, idx] == 0
        H[dead, dead] = 1.0
        H[idx, idx] += damping * H[idx, idx].mean()
        return H

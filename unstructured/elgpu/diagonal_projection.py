"""The Z-step: diagonal-weighted magnitude projection.

Given the target quadratic form (z - v)^T H (z - v), a full-Hessian
projection needs to invert H, which gets expensive and numerically
fragile at the sparsity levels we run at. Following ELSA's own approach,
we use the diagonal approximation instead:

    sum_i H_ii * (z_i - v_i)^2

Because this is separable across coordinates, the optimal k-sparse
solution has a closed form with no matrix inversion at all: for any
coordinate you decide to keep, the zero-cost choice is z_i = v_i
(leave it untouched); the cost of zeroing a coordinate instead is
H_ii * v_i^2. So the best k-sparse projection just keeps the k
coordinates with the largest H_ii * v_i^2 score in each row and zeros
the rest.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def diagonal_project(V: torch.Tensor, H: torch.Tensor, k_keep: int) -> torch.Tensor:
    """Project V (d_out, d_in) onto "k_keep nonzeros per row".

    Args:
        V: matrix to project (v = x + u in the ADMM update).
        H: damped layer Hessian; only its diagonal is used.
        k_keep: nonzeros to keep per row, same for every row.

    Returns:
        Z, same shape as V: exactly k_keep nonzeros per row, each equal
        to its original V value (survivors are never rescaled).
    """
    diag = torch.diagonal(H).clamp_min(0.0)
    score = V.pow(2) * diag.unsqueeze(0)
    keep_idx = torch.topk(score, k_keep, largest=True, dim=1).indices
    Z = torch.zeros_like(V)
    Z.scatter_(1, keep_idx, V.gather(1, keep_idx))
    return Z

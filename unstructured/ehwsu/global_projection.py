"""Phase 2's Z-step for the unstructured variant: one global top-k across
every prunable layer at once, instead of giving each layer the same fixed
sparsity ratio.

The projection objective only ever asks for a total nonzero *count*
budget across the whole network -- nothing requires that budget to be
split evenly across layers. This module drops that split-evenly
assumption: it scores every weight the same way the per-layer Z-step
does (`H_ii * v_i^2`, the cost of forcing that coordinate to zero), pools
every layer's scores together, and keeps the globally largest
`k_keep_total` of them. A layer whose weights are collectively more
salient keeps proportionally more of the budget; a redundant layer gives
up proportionally more.

**Why the scores need to be normalized per layer first.** Pooling raw
scores across layers with no adjustment doesn't work: OPT (like GPT-2)
uses residual-scaled initialization, so the second linear layer in each
attention/MLP sub-block (`out_proj`, `fc2`) starts with deliberately
smaller weight variance than the first (`q/k/v_proj`, `fc1`) -- purely to
control how fast the residual stream's variance grows with depth. That
init-scale difference alone is enough to make `out_proj`/`fc2`'s raw
scores read as "globally unimportant" everywhere, regardless of whether
they actually are: an early, unnormalized version of this function
pruned those layers to 80%+ average sparsity (up to 96.6% in one layer)
while leaving `q_proj`/`k_proj` near 15%, uniformly across every block --
the signature of an initialization artifact, not a real importance
signal, and it produced a WikiText2 perplexity of 1573 (dense: 27.66)
under otherwise identical settings to a healthy run.

The fix is to divide each layer's raw scores by that layer's own mean
before pooling. This cancels a uniform per-layer scale factor (exactly
the residual-init artifact above) while preserving each layer's
*within-layer* relative ranking and any genuine difference in how
concentrated a layer's importance actually is.
"""

from __future__ import annotations

import torch


def _raw_scores(V: dict[str, torch.Tensor], H: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-layer H_ii * v_i^2 -- the cost of forcing each coordinate to zero."""
    scores = {}
    for name, v in V.items():
        diag = torch.diagonal(H[name]).clamp_min(0.0)  # (d_in,)
        scores[name] = v.pow(2) * diag.unsqueeze(0)  # (d_out, d_in)
    return scores


@torch.no_grad()
def global_diagonal_project(
    V: dict[str, torch.Tensor],
    H: dict[str, torch.Tensor],
    k_keep_total: int,
) -> dict[str, torch.Tensor]:
    """Project every layer's v_l = x_l + u_l onto one shared, global
    "k_keep_total nonzeros total" constraint.

    Args:
        V: {layer_name: (d_out, d_in) tensor}, one entry per prunable
            layer, all on the same device.
        H: {layer_name: (d_in, d_in) damped Hessian}, same keys as V
            (only each Hessian's diagonal is used).
        k_keep_total: total nonzeros to keep, summed across every layer's
            every row -- the single free parameter. No per-layer or
            per-row count is specified anywhere else.

    Returns:
        {layer_name: Z}, each Z the same shape as the matching V, such
        that the total nonzero count across every Z equals
        k_keep_total exactly (ties at the threshold are broken
        deterministically). Individual layers' nonzero fractions are
        free to differ.
    """
    raw = _raw_scores(V, H)
    normalized = {name: s / s.mean().clamp_min(1e-30) for name, s in raw.items()}

    names = list(V.keys())
    if not names:
        return {}

    flat_scores = torch.cat([normalized[name].reshape(-1) for name in names])
    n_total = flat_scores.numel()
    k_keep_total = max(0, min(k_keep_total, n_total))

    if k_keep_total == 0:
        return {name: torch.zeros_like(V[name]) for name in names}
    if k_keep_total >= n_total:
        return {name: V[name].clone() for name in names}

    # Find the k_keep_total-th largest score via the (n-k+1)-th smallest,
    # rather than materializing a global top-k index tensor -- at, say,
    # 50% of a 1.3B-parameter model that would need ~650M int64 indices,
    # a few GB on its own.
    threshold = torch.kthvalue(flat_scores, n_total - k_keep_total + 1).values
    mask_flat = flat_scores >= threshold

    n_kept = int(mask_flat.sum().item())
    if n_kept > k_keep_total:
        # Ties only really happen at exactly zero (dead input channels
        # legitimately share a score of 0.0). Break them deterministically
        # by dropping the first few tied entries in flat order, so the
        # kept count lands exactly on k_keep_total.
        tie_idx = (flat_scores == threshold).nonzero(as_tuple=True)[0]
        n_excess = n_kept - k_keep_total
        mask_flat[tie_idx[:n_excess]] = False

    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        v = V[name]
        n = v.numel()
        layer_mask = mask_flat[offset : offset + n].view_as(v)
        out[name] = torch.where(layer_mask, v, torch.zeros_like(v))
        offset += n
    return out

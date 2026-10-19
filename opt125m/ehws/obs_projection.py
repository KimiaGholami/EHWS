"""An alternative Z-step: full-Hessian-aware OBS projection (SparseGPT-stable variant).

Solves, per Phase 1 slide 39 (Eq. 2) / Phase 2 slide 41 (Eq. 7):

    z_l^{t+1} = argmin_{||z_l||_0 <= k_l}  (z_l - v_l)^T H_l (z_l - v_l)

i.e. project ``v = x + u`` onto the k-sparse-per-row constraint set under
the Mahalanobis-like distance induced by the layer's Hessian ``H_l``.

**No longer the production default** (see `ehws/diagonal_projection.py`,
which is): slide 39 states the Z-step only as this objective, without
mandating a solver, and the user explicitly chose to match real ELSA's
own Diag(H) simplification instead of this module's exact full-Hessian
solve. This module remains available as the injectable alternative
(``zstep_fn=obs_project`` at either `admm.run_phase1`/`run_phase2` call
site) for controlled ablations isolating whether full-Hessian-awareness
itself changes results.

Implementation note (deliberate deviation from a literal reading of slide
30): slide 30's incremental per-row *adaptive*-order update tracks only a
running scalar diagonal per column to avoid literally re-inverting the
Hessian at every single greedy removal. That is only accurate for a
handful of sequential eliminations -- verified against an independent
ground truth (re-inverting the true reduced submatrix at each step), it
silently drifts and then diverges (the tracked "diagonal" runs negative,
corrections blow up to inf/NaN) after 30-200+ sequential removals, well
short of the ~5000 removals-per-row this method needs at 90% sparsity on
its larger layers (see PROGRESS.md for the numerical experiments behind
this decision, and the three options weighed).

Given that, this module reuses SparseGPT's own proven-stable mechanics:
each row independently ranks its own candidate columns via the standard
diagonal OBS score (``w^2 / Hinv_jj``), computed fresh per
``blocksize=128`` column block. Row selection is genuinely row-specific:
different rows can and do prune different column positions. The
*correction* for a block's own surviving weights is computed exactly,
using only that block's own ``H_bb`` submatrix (a small batched linear
solve over each row's own alive/dead partition within the block).

**Cross-block propagation was attempted and reverted -- do not
re-attempt it without reading this note.** A version of this module
tried marginalizing each block's dead columns out of every later
column's target, via the exact closed-form group-OBS formula
``v_future -= H[future,dead] @ H[dead,dead]^-1 @ v[dead]``. That formula
is mathematically correct for the sub-problem it solves (verified
against an independent from-scratch reference, diff ~1e-9), and it
passed unit tests built on random well-conditioned synthetic Hessians.
It was still wrong to ship: on REAL OPT-125M calibration Hessians
(condition numbers ~1e4-3e4, measured directly), it produced a relative
output-reconstruction error of 57% at just 10% sparsity -- vs. 1% for
*plain magnitude pruning with no Hessian at all* on the same layer -- and
this alone was enough to make full end-to-end perplexity catastrophic
(PPL in the thousands) at every sparsity level tested, including ones as
mild as 10%. The mechanism: the columns selected as "dead" are, by
construction, the least individually important ones -- and in real LLM
activations, the least-important channels are frequently near-duplicates
or highly correlated with each other (that redundancy is *why* they're
prunable). That makes ``H[dead,dead]`` frequently near-singular, so
inverting it to compute the marginalization amplifies noise into a huge,
wrong correction that then contaminates every later block. Random
synthetic test matrices don't reliably reproduce that correlation
structure, which is why the unit tests passed while real data broke
catastrophically -- a concrete, measured instance of the general lesson
"exact-for-a-well-posed-subproblem" is not the same guarantee as
"numerically safe on real, correlated data." If cross-block awareness is
revisited, it needs a real-data reconstruction-error check (see
`tests/test_obs_projection.py`'s synthetic tests are NOT sufficient on
their own) before it can be trusted again, and almost certainly needs
either a much larger, correlation-aware ridge on the dead-set solve
specifically, or an entirely different mechanism.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def obs_project(V: torch.Tensor, H: torch.Tensor, k_keep: int, blocksize: int = 128, eps: float = 1e-6) -> torch.Tensor:
    """Project ``V`` (d_out, d_in) onto "k_keep nonzeros per row" under H.

    Args:
        V: (d_out, d_in) matrix to project (``v_l = x_l + u_l`` in the
            ADMM Z-step).
        H: (d_in, d_in) damped layer Hessian (NOT its inverse).
        k_keep: nonzeros to keep per row (uniform across rows).
        blocksize: column-block width for the local exact solve (128,
            matching SparseGPT/CWS convention).
        eps: extra ridge added before each block-local solve, purely for
            floating-point safety margin beyond the caller's own damping.

    Returns:
        Z: (d_out, d_in), the OBS-corrected surviving values, exactly
           k_keep nonzeros per row (up to per-block integer rounding,
           the same convention SparseGPT/CWS use).
    """
    d_out, d_in = V.shape
    device, dtype = V.device, torch.float64
    V = V.to(dtype)
    H = H.to(dtype)

    Z = torch.zeros((d_out, d_in), device=device, dtype=dtype)
    n_blocks = math.ceil(d_in / blocksize)

    for b in range(n_blocks):
        start = b * blocksize
        end = min(start + blocksize, d_in)
        B = end - start
        # Round-to-nearest on the *keep* count, not floor() on a
        # subtracted sparsity fraction -- floor(B*(1-k_keep/d_in)) silently
        # off-by-ones whenever floating point rounds B*k_keep/d_in a hair
        # below its true integer value (e.g. 12*5/12 landing on
        # 4.999999999999999 -- verified this actually happens).
        k_keep_block = round(B * k_keep / d_in)
        k_prune = B - k_keep_block
        v_block = V[:, start:end]  # (d_out, B)

        if k_prune <= 0:
            Z[:, start:end] = v_block
            continue
        if k_prune >= B:
            continue  # whole block pruned; Z already zero there

        H_block = H[start:end, start:end]
        L = torch.linalg.cholesky(H_block)
        Hinv_block = torch.cholesky_inverse(L)
        diag = torch.diagonal(Hinv_block).clamp_min(1e-12)

        # Per-row diagonal OBS score (SparseGPT's own selection formula,
        # slide 18: score[j] = w[j]^2 / Hinv[j,j]) -- already row-specific
        # via v_block, the denominator is the only shared part.
        score = v_block.pow(2) / diag.unsqueeze(0)
        dead_idx = torch.topk(score, k_prune, largest=False, dim=1).indices  # (d_out, k_prune)
        alive_mask = torch.ones((d_out, B), dtype=torch.bool, device=device)
        alive_mask.scatter_(1, dead_idx, False)
        n_alive = B - k_prune
        alive_idx = alive_mask.nonzero(as_tuple=False)[:, 1].view(d_out, n_alive)

        # Exact joint OBS correction for THIS row's specific dead set,
        # using the full off-diagonal block Hessian (not a diagonal-only
        # or sequential approximation):
        #   z[alive] = v[alive] - H[alive,alive]^-1 @ H[alive,dead] @ v[dead]
        H_AA = H_block[alive_idx.unsqueeze(2), alive_idx.unsqueeze(1)]  # (d_out, n_alive, n_alive)
        H_AD = H_block[alive_idx.unsqueeze(2), dead_idx.unsqueeze(1)]  # (d_out, n_alive, k_prune)
        v_dead = v_block.gather(1, dead_idx)  # (d_out, k_prune)
        v_alive = v_block.gather(1, alive_idx)  # (d_out, n_alive)

        ridge = eps * torch.diagonal(H_AA, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
        eye = torch.eye(n_alive, device=device, dtype=dtype).unsqueeze(0)
        H_AA = H_AA + ridge.unsqueeze(-1) * eye

        rhs = torch.bmm(H_AD, v_dead.unsqueeze(-1))  # (d_out, n_alive, 1)
        correction = torch.linalg.solve(H_AA, rhs).squeeze(-1)  # (d_out, n_alive)
        z_alive = v_alive - correction

        z_block = torch.zeros((d_out, B), device=device, dtype=dtype)
        z_block.scatter_(1, alive_idx, z_alive)
        Z[:, start:end] = z_block

    return Z.to(V.dtype)


@torch.no_grad()
def obs_select(V: torch.Tensor, H: torch.Tensor, k_keep: int, blocksize: int = 128, eps: float = 1e-6) -> torch.Tensor:
    """Same per-row OBS selection as ``obs_project`` (score = w^2/Hinv_jj,
    full off-diagonal block Hessian, row-specific dead sets), but with
    ELSA's own no-correction simplification applied to the survivors
    instead of this module's exact joint correction: ``z[alive] = v[alive]``
    unchanged, not ``v[alive] - H_AA^-1 @ H_AD @ v[dead]``.

    This isolates *selection* quality from *correction* in a controlled
    ablation against both ``diagonal_project`` (diagonal score, no
    correction) and ``obs_project`` (OBS score, full correction) -- the
    2x2 this module's docstring and ``diagonal_project``'s docstring both
    describe. Still costs one Cholesky/inverse per (blocksize, blocksize)
    block (same as ``obs_project``'s selection half) -- only the
    per-row alive/alive linear solve is skipped, which is a small
    fraction of ``obs_project``'s total cost since that solve is
    batched over rows.
    """
    d_out, d_in = V.shape
    device, dtype = V.device, torch.float64
    V = V.to(dtype)
    H = H.to(dtype)

    Z = torch.zeros((d_out, d_in), device=device, dtype=dtype)
    n_blocks = math.ceil(d_in / blocksize)

    for b in range(n_blocks):
        start = b * blocksize
        end = min(start + blocksize, d_in)
        B = end - start
        k_keep_block = round(B * k_keep / d_in)
        k_prune = B - k_keep_block
        v_block = V[:, start:end]  # (d_out, B)

        if k_prune <= 0:
            Z[:, start:end] = v_block
            continue
        if k_prune >= B:
            continue  # whole block pruned; Z already zero there

        H_block = H[start:end, start:end]
        L = torch.linalg.cholesky(H_block)
        Hinv_block = torch.cholesky_inverse(L)
        diag = torch.diagonal(Hinv_block).clamp_min(1e-12)

        score = v_block.pow(2) / diag.unsqueeze(0)
        dead_idx = torch.topk(score, k_prune, largest=False, dim=1).indices  # (d_out, k_prune)
        alive_mask = torch.ones((d_out, B), dtype=torch.bool, device=device)
        alive_mask.scatter_(1, dead_idx, False)

        z_block = v_block * alive_mask.to(dtype)  # survivors unchanged, dead zeroed -- no correction
        Z[:, start:end] = z_block

    return Z.to(V.dtype)

"""The Z-step: real ELSA's diagonal-weighted magnitude projection.

Real ELSA (`lib/optimizers.py` in the official repo, matching Eq. 10-11 of
the paper) does NOT use a full-Hessian OBS-style projection. It replaces
the intractable-in-general quadratic form ``(z-v)^T H (z-v)`` with the
*diagonal* approximation ``sum_i H_ii (z_i - v_i)^2`` specifically for
numerical simplicity -- the paper says so directly (Section 3.2:
"We found that using Diag(H) allows us to retain this simplicity").

Because the diagonal form is separable across coordinates, the exact
solution has a trivial closed form with no matrix inversion anywhere:
for a fixed choice of which ``k`` entries survive, the zero-cost choice
for a surviving entry is ``z_i = v_i`` (untouched), and the cost of
forcing a coordinate to zero is ``H_ii * v_i^2``. So the optimal k-sparse
projection just keeps the ``k`` entries with the largest ``H_ii * v_i^2``
score and zeroes everything else -- no correction, no inversion, no
possible numerical blow-up from an ill-conditioned Hessian block.

**This is the production default Z-step** (`admm.run_phase1` /
`admm.run_phase2`'s ``zstep_fn``), by the user's explicit choice: slide
39 states the Z-step only as the objective ``(z-v)^T H (z-v)``, without
mandating a specific solver, and three readings were on the table --
(1) CWS's slide-30 incremental r/d recursion (a different, unstable
method from this deck's Section 3, not something real ELSA does), (2)
this module, matching real ELSA's own Diag(H) simplification, or (3) an
exact full-Hessian solver (`ehws.obs_projection.obs_project`). The user
chose (2) to match real ELSA exactly. `obs_project` remains available as
the injectable alternative for controlled ablations (mirroring ELSA's own
Table 3 methodology of turning the projection's Hessian-awareness on and
off) -- swap it back in via `zstep_fn=obs_project` to isolate whether the
full-Hessian correction changes the observed OPT-125M divergence.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def diagonal_project(V: torch.Tensor, H: torch.Tensor, k_keep: int, **_ignored) -> torch.Tensor:
    """Project ``V`` (d_out, d_in) onto "k_keep nonzeros per row", ELSA-style.

    Args:
        V: (d_out, d_in) matrix to project (``v_l = x_l + u_l``).
        H: (d_in, d_in) damped layer Hessian (only its diagonal is used).
        k_keep: nonzeros to keep per row (uniform across rows).
        **_ignored: absorbs extra kwargs (e.g. ``blocksize``) so this can
            be swapped in for ``obs_project`` with the same call site.

    Returns:
        Z: (d_out, d_in), exactly k_keep nonzeros per row, each equal to
           its original ``V`` value (no correction applied to survivors).
    """
    diag = torch.diagonal(H).clamp_min(0.0)  # (d_in,), Fhat_ii in ELSA's Eq. 11
    # Score computed in float32 even when V/H are stored in bf16 (large
    # models only fit in memory at all with weights/state in bf16 -- see
    # run_pipeline.py's --dtype) -- bf16's ~7-8 mantissa bits are coarse
    # enough that many close-scoring weights would round to identical
    # scores, turning "keep the top-k" into an arbitrary tie-break among
    # them. The transient float32 score tensor is freed right after
    # topk, so this costs no persistent memory, only a brief spike.
    score = V.float().pow(2) * diag.float().unsqueeze(0)  # (d_out, d_in)
    keep_idx = torch.topk(score, k_keep, largest=True, dim=1).indices
    Z = torch.zeros_like(V)
    Z.scatter_(1, keep_idx, V.gather(1, keep_idx))
    return Z


@torch.no_grad()
def magnitude_project(V: torch.Tensor, H: torch.Tensor, k_keep: int, **_ignored) -> torch.Tensor:
    """Same as `diagonal_project` but with the Hessian weighting dropped:
    score is plain ``v_i^2`` (equivalently |v_i|, same top-k ordering),
    with no ``H_ii`` term at all -- real ELSA's *default* Z-step
    (`admm_projection_mode="identity"`) has no Hessian involved anywhere,
    unlike `diagonal_project` above. Ablation to isolate whether
    `diagonal_project`'s Hessian weighting (built on real-data Hessians
    documented elsewhere in this codebase as ill-conditioned,
    condition numbers ~1e4-3e4) is itself hurting HGRN, independent of
    the `--damping` ridge-term experiments which keep the Hessian
    weighting but try to make it better-behaved.
    """
    score = V.float().pow(2)  # (d_out, d_in), no H_ii term
    keep_idx = torch.topk(score, k_keep, largest=True, dim=1).indices
    Z = torch.zeros_like(V)
    Z.scatter_(1, keep_idx, V.gather(1, keep_idx))
    return Z

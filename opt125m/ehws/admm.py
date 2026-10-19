"""Phase 1 (layer-wise ADMM) and Phase 2 (global ADMM) drivers.

Phase 1, per layer l, sequentially (Eq. 1-4):
    x-step:  x_l <- argmin_x  f(x_l) + (lambda/2) ||x_l - z_l + u_l||_F^2
    z-step:  z_l <- argmin_{||z_l||_0<=k_l} (z_l - v_l)^T H_l (z_l - v_l)
    u-step:  u_l <- u_l + x_l^{t+1} - z_l^{t+1}
One continuous ADMM run per layer straight at `cfg.target_sparsity`,
penalty cosine-ramped 0 -> admm_lambda_max, matching
`Extreme_Layer_Global_Pruning_Unstructured`'s Phase 1 shape rather than
this package's own original saturation-gated ladder -- see this module's
"Why Phase 1 no longer has a ladder" note below for why that was dropped.

Phase 2 (Eq. 6-7): every layer's x_l becomes simultaneously trainable
under one *global* forward pass (letting gradients couple across layers),
warm-started from Phase 1's {x_l, z_l, u_l, H_l}, with a single uniform
target sparsity ratio `s` shared by every layer (k_l = floor((1-s)*N_l)).

## Why Phase 1 no longer has a ladder

The original design here climbed a per-layer sparsity ladder (15%, 30%,
..., 95%), gated by a saturation check that stopped a layer once it
seemed to be hurting more than helping. Two different checks were tried
and both are documented, real-calibration-validated failures (see
PROGRESS.md for the full history):

1. The literal dual-residual check from the design doc
   (`||u_l|| > tau * ||x_l^0||`): measured directly against OPT-125M's
   real weights to be close to scale-invariant -- pretrained transformer
   weights are roughly Gaussian per layer, and the L2-mass fraction in
   the bottom-k% of a Gaussian barely depends on that layer's own scale.
   Every layer saturated at nearly the same rung regardless of type or
   depth, which is the check measuring weight-distribution shape, not
   actual loss sensitivity.
2. Gating directly on the true CE+KD loss instead (seemingly the more
   principled fix) was implemented, then validated on real calibration
   data (`results/opt-125m-lossgate-validation/`) and made Phase 2's
   final perplexity *worse* than the scale-invariant check it replaced
   (132.42 vs. 78.10 WikiText2 PPL at 50% sparsity) -- the check turned
   out to be too noisy (an 8-sequence batch deciding each layer's fate)
   and too myopic (no accounting for 72 individually-small layer
   decisions compounding together) to trust.

Neither check is salvageable without more work than this package has
budget for right now. `Extreme_Layer_Global_Pruning_Unstructured`
(EHWS-U) sidesteps the whole question: instead of searching for each
layer's own tolerable sparsity via a ladder, it just runs one continuous
ADMM pass per layer straight at a single fixed target (0.7, chosen by an
offline sweep over {0.3, 0.5, 0.7, 0.9} against a downstream Phase 2
target of 0.5 -- 0.7 won, 0.9 was worst), with the ADMM penalty
cosine-ramped instead of a discrete ladder. That's the design this module
now uses too. It is *not* independently re-swept for this package's
uniform-ratio Phase 2 (only EHWS-U's own global-budget Phase 2 was used
to pick 0.7) -- carried over as the best validated starting point
available, not a value tuned for this specific pairing.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import os

import torch
import torch.nn as nn

from .diagonal_projection import diagonal_project
from .hessian import LayerHessian
from .losses import combined_loss
from .model_layers import PrunableLayer
from .obs_projection import obs_project

# Mixed precision for the student's forward passes (`--master-fp32`): weights,
# grads and Adam state stay float32 (master copy) and matmuls run in this dtype
# under autocast -- what ELSA's HF Trainer does with `bf16=True` on an fp32
# model. None = no autocast (the model runs in its own storage dtype).
_AUTOCAST_DTYPE: torch.dtype | None = None


def set_autocast_dtype(dtype: torch.dtype | None) -> None:
    global _AUTOCAST_DTYPE
    _AUTOCAST_DTYPE = dtype


def _student_autocast(device) -> contextlib.AbstractContextManager:
    if _AUTOCAST_DTYPE is None:
        return contextlib.nullcontext()
    # cache_enabled=False: don't keep a bf16 copy of every weight alive for the
    # whole forward (~2.7GB on HGRN-1.3B); casts are redone per use instead.
    return torch.autocast(torch.device(device).type, dtype=_AUTOCAST_DTYPE, cache_enabled=False)


class EpochSampler:
    """Draw calibration batches without replacement, reshuffling after each pass.

    ELSA's Trainer walks its 32,768 training sequences once in random order;
    `_sample_batch` draws with replacement instead. With 4096 steps x 8
    sequences this sampler sees each of 32,768 sequences exactly once.
    """

    def __init__(self, calib_ids: torch.Tensor, seed: int = 0):
        self.calib_ids = calib_ids
        self.gen = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(calib_ids.shape[0], generator=self.gen)
        self.pos = 0

    def __call__(self, n: int, device) -> torch.Tensor:
        if self.pos + n > len(self.perm):
            self.perm = torch.randperm(self.calib_ids.shape[0], generator=self.gen)
            self.pos = 0
        idx = self.perm[self.pos : self.pos + n]
        self.pos += n
        return self.calib_ids[idx].to(device)


@dataclasses.dataclass
class LayerState:
    name: str
    module: nn.Linear
    x: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    H: torch.Tensor
    n_in: int
    n_out: int
    x0_norm: float
    final_sparsity: float = 0.0
    # Penalty in effect when `u` was last updated. `u` is ADMM's *scaled*
    # dual (the real dual is admm_lambda * u), so handing `u` to a phase
    # with a different penalty changes the dual's actual strength unless
    # it's rescaled -- see Phase2Config.dual_init. 0.0 = no dual yet.
    admm_lambda: float = 0.0


def save_phase1_checkpoint(states: dict[str, "LayerState"], path: str, meta: dict) -> None:
    """Persist Phase 1's (or build_dense_states') output to disk.

    Motivated by a real loss: at HGRN-1.3B scale, Phase 1 took ~4.5 hours
    and Phase 2 then OOM'd immediately on entering training (a real,
    separate activation-memory bug, since fixed) -- with nothing saved,
    that whole 4.5 hours had to be redone from scratch just to retry
    Phase 2. `module` isn't saved (not serializable, not needed -- reload
    re-associates by layer name against a freshly `discover_prunable_layers`'d
    model instead, see `load_phase1_checkpoint`). Every tensor moves to
    CPU first regardless of where it started, both to keep the checkpoint
    file portable and to avoid holding an extra GPU-resident copy during
    the (potentially slow, for a 1.3B+ model) save.

    `H` is deliberately *not* saved. It's the single biggest tensor per
    layer (a dense d_in x d_in matrix -- e.g. ~11GB total across
    HGRN-1.3B's 168 layers even at float32, per `_compute_hessian`'s
    caller), and it's fully deterministic given the calibration data and
    each layer's already-known committed weight -- see
    `load_phase1_checkpoint`, which re-derives it instead. A real incident
    prompted this: a 4.4GB `phase1_checkpoint.pt.tmp` was still being
    written to the NFS-mounted home filesystem when its job got killed,
    losing the run and leaving an orphaned partial file -- H alone was
    most of that size. `z` is saved sparse (`.to_sparse()`) since at these
    target sparsities it's 70-95% zeros by construction (`diagonal_project`
    always returns exact zeros, never near-zero noise), so this is a
    lossless size cut, not an approximation.
    """
    payload = {
        "meta": meta,
        "layers": {
            name: {
                "x": s.x.cpu(), "z": s.z.cpu().to_sparse(), "u": s.u.cpu(),
                # Phase 1 trains each layer's bias along with its weight (as real
                # ELSA does -- its optimizer trains every non-frozen parameter,
                # ADMM only constrains the weights), so the live bias is part of
                # Phase 1's output too.
                "bias": None if s.module.bias is None else s.module.bias.detach().cpu().clone(),
                "n_in": s.n_in, "n_out": s.n_out, "x0_norm": s.x0_norm,
                "final_sparsity": s.final_sparsity, "admm_lambda": s.admm_lambda,
            }
            for name, s in states.items()
        },
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on the same filesystem -- a crash mid-save can't leave a corrupt checkpoint at `path`


def load_phase1_checkpoint(
    path: str, layers: list["PrunableLayer"], device, meta: dict,
    model: nn.Module, calib_ids: torch.Tensor, seqlen: int, damping: float,
    log_fn=print, hessian_n_calib: int | None = None, diag_only: bool = False,
) -> dict[str, "LayerState"]:
    """Load a checkpoint saved by `save_phase1_checkpoint`, re-associating
    each saved layer's tensors with the corresponding (freshly-loaded)
    module by name, and restoring that module's live weight to `x`
    (matching what `run_phase1`/`build_dense_states` leave live at the
    end of their own run).

    `H` isn't in the checkpoint (see `save_phase1_checkpoint`), so it's
    recomputed here via the same `_compute_hessian` call `run_phase1`/
    `build_dense_states` used originally, replayed in the same layer
    order with each earlier layer's weight set to its saved (committed)
    `z` first -- exactly reproducing the partially-pruned model state
    each layer's H was originally measured against. This costs one
    calibration forward pass per layer (what computing H always costs),
    not a re-run of Phase 1's ADMM optimization -- the expensive part
    this checkpoint exists to avoid redoing.

    `meta` must match what the checkpoint was saved with exactly --
    refuses to silently load a checkpoint from a different model/config
    into the wrong run (e.g. a stale checkpoint left over from a
    different `--model` or `--p1-target-sparsity` reusing the same
    `--out` directory). It also gates correctness of the H recompute
    above: `n_calib`/`seqlen`/`seed`/`damping` all affect the Hessian
    value, so a checkpoint saved under different values of those must be
    rejected rather than silently reused.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_meta = payload.get("meta", {})
    if saved_meta != meta:
        raise ValueError(
            f"checkpoint at {path} was saved with meta={saved_meta!r}, but this run's meta is {meta!r} -- "
            f"refusing to load a checkpoint from a different run (delete the checkpoint file if this is intentional)"
        )
    by_name = {pl.name: pl for pl in layers}
    states: dict[str, LayerState] = {}
    for name, d in payload["layers"].items():
        pl = by_name[name]
        state = LayerState(
            name=name, module=pl.module,
            x=d["x"].to(device), z=d["z"].to_dense().to(device), u=d["u"].to(device),
            H=None,
            n_in=d["n_in"], n_out=d["n_out"], x0_norm=d["x0_norm"], final_sparsity=d["final_sparsity"],
            admm_lambda=d.get("admm_lambda", 0.0),
        )
        states[name] = state

    log_fn(f"Recomputing {len(layers)} layer Hessian(s) (not stored in the checkpoint -- see save_phase1_checkpoint)")
    for li, pl in enumerate(layers):
        s = states[pl.name]
        # `pl.module.weight` is still this layer's original dense weight here
        # (untouched by this loop so far -- only earlier layers were committed
        # to `z` below) -- exactly the state `run_phase1`/`build_dense_states`
        # measured H against originally, before this layer's own ADMM run.
        s.H = _compute_hessian(
            model, pl.module, calib_ids, seqlen, device, damping,
            n_samples=hessian_n_calib, diag_only=diag_only,
        ).to(device)
        pl.module.weight.data.copy_(s.z)  # commit to the pruned value before the next layer's forward pass measures its H
        # Restore the trained bias too: run_phase1 measured every later layer's H
        # with this layer's bias already trained. Checkpoints saved before biases
        # were stored have no "bias" key and keep the dense bias (the old behavior).
        bias = payload["layers"][pl.name].get("bias")
        if bias is not None and pl.module.bias is not None:
            pl.module.bias.data.copy_(bias.to(device=pl.module.bias.device, dtype=pl.module.bias.dtype))
        log_fn(f"  [{li+1}/{len(layers)}] {pl.name}: Hessian recomputed")

    for pl in layers:
        # A fresh (non-checkpointed) run_phase1 leaves each module's live weight
        # at `z` (its final `module.weight.data.copy_(z)`), not `x` -- z is the
        # actual pruned result; x is only its pre-projection ADMM value, and the
        # two differ except in build_dense_states' cold-start case (x0==x==z
        # there). Matters for --skip-phase2, which evaluates directly off
        # whatever's live here without going through run_phase2 (which would
        # otherwise overwrite it with x itself regardless).
        pl.module.weight.data.copy_(states[pl.name].z)
    return states


@dataclasses.dataclass
class Phase1Config:
    # Fixed per-layer warm-start target, independent of whatever overall
    # sparsity Phase 2 is eventually asked for -- Phase 2's own Z-step
    # re-derives its own k_keep from scratch regardless of this value, so
    # Phase 1 runs once per model and gets reused across every Phase 2
    # sparsity level. Carried over from EHWS-U's own validated sweep
    # (see module docstring) -- not independently re-swept here.
    target_sparsity: float = 0.7
    # rounds * x_steps = 56 total optimizer steps per layer (EHWS-U's
    # own validated values).
    rounds: int = 14
    x_steps: int = 4
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    admm_lambda_max: float = 5e-5
    # "cosine" (ramp 0 -> admm_lambda_max, original behavior) or "constant".
    # 5e-5/cosine is the same value found 200x too small for HGRN's Phase 2
    # in Session 5 -- never revisited for Phase 1. See PROGRESS.md Session 10.
    lambda_schedule: str = "cosine"
    alpha_kd: float = 0.5
    kd_temperature: float = 1.0  # 1.0 = untempered KD (original behavior); see losses.py's combined_loss
    damping: float = 0.01
    # Calibration sequences used per Hessian (None = all of them). See _compute_hessians.
    hessian_n_calib: int | None = None
    diag_only_hessian: bool = False
    seqlen: int = 2048
    max_grad_norm: float = 1.0  # matches real ELSA's clip_grad_norm_ default (HF TrainingArguments convention)
    beta1: float = 0.9
    beta2: float = 0.999  # PyTorch's Adam stock default; real ELSA sets 0.95 (ELSA-official/lib/trainer.py's admm_beta2) -- see run_pipeline.py's --adam-beta2
    prox_after_clip: bool = True  # add the ADMM penalty gradient after clip_grad_norm_ instead of inside the backpropped loss, matching real ELSA's _proximal_update (see _x_step's docstring). Made default 2026-09-11 -- beat prior default on both WT2/C4 in the opt-125m-fixablation ablation, see PROGRESS.md


@dataclasses.dataclass
class Phase2Config:
    target_sparsity: float = 0.5
    # rounds*x_steps = 4096 total optimizer steps, x_steps=32 = the ADMM
    # dual/z update interval -- both match real ELSA's Table 4 exactly
    # ("Training steps: 4096", "Interval k: 32"). The previous defaults
    # (rounds=10, x_steps=5 => 50 total steps) were ~82x short of this,
    # found while comparing this repo's fresh OPT-125M run against ELSA's
    # own published Table 6 numbers for the same model (see PROGRESS.md).
    rounds: int = 128
    x_steps: int = 32
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 1e-4
    admm_lambda_max: float = 5e-5
    lambda_schedule: str = "cosine"  # "cosine" (ramp 0->max, ELSA's OPT-1.3B+ convention) or "constant" (ELSA's OPT-125M/Gemma-2-2B convention)
    # How Phase 1's scaled dual `u` enters Phase 2:
    #   "carry"   -- as-is (original behavior). Phase 1's penalty (5e-5 by
    #                default) is far below Phase 2's, so the real dual
    #                (lambda * u) jumps by lambda_p2/lambda_p1 at the handoff.
    #   "rescale" -- u *= lambda_p1 / admm_lambda_max, keeping the real dual
    #                unchanged across the handoff. Made default 2026-09-26 --
    #                beat carry and zero on both WT2/C4 in the Session 10
    #                OPT-125M sweep, see PROGRESS.md
    #   "zero"    -- u = 0; Phase 2 keeps Phase 1's x and z but starts a fresh dual.
    dual_init: str = "rescale"
    alpha_kd: float = 0.5
    kd_temperature: float = 1.0  # 1.0 = untempered KD (original behavior); see losses.py's combined_loss
    seqlen: int = 2048
    max_grad_norm: float = 1.0  # matches real ELSA's clip_grad_norm_ default (HF TrainingArguments convention)
    beta1: float = 0.9
    beta2: float = 0.999  # PyTorch's Adam stock default; real ELSA sets 0.95 (ELSA-official/lib/trainer.py's admm_beta2) -- see run_pipeline.py's --adam-beta2
    prox_after_clip: bool = True  # add the ADMM penalty gradient after clip_grad_norm_ instead of inside the backpropped loss, matching real ELSA's _proximal_update (see _x_step's docstring). Made default 2026-09-11 -- beat prior default on both WT2/C4 in the opt-125m-fixablation ablation, see PROGRESS.md
    # Storage dtype for z and u (None = the weight's own dtype). ELSA's HGRN run
    # kept its split/dual in bf16 (--admm_split_dtype/--admm_dual_dtype bf16).
    state_dtype: torch.dtype | None = None
    # Also train every non-pruned parameter (norms, embeddings, lm_head, HGRN's
    # forget-gate lower bounds) with the same Adam/lr, no ADMM penalty -- what
    # ELSA's create_optimizer does ("other_params" group).
    train_extra: bool = False
    # Re-measure diag(H) for the Z-step from the x-step's own student forwards
    # (inputs of every prunable layer, accumulated since the previous Z-step),
    # so the Hessian score tracks the network as it's pruned. No extra passes.
    hessian_refresh: bool = False
    # 0 = replace diag(H) with each window's measurement; beta in (0, 1) =
    # H <- beta * H + (1 - beta) * H_window, to damp round-to-round drift.
    hessian_ema: float = 0.0
    # Which non-pruned parameters `train_extra` trains: "all" (ELSA's choice)
    # or "norms" (LayerNorm weights/biases only; leaves the tied
    # embedding/lm_head frozen).
    train_extra_scope: str = "all"
    damping: float = 0.01  # same ridge as the initial Hessian, applied to refreshed diagonals
    # Draw Phase 2 batches without replacement (EpochSampler) instead of with.
    epoch_sampling: bool = False
    seed: int = 0


def _k_keep(n_in: int, sparsity: float) -> int:
    """Nonzeros to keep per row for a given fraction pruned.

    Rounds the *pruned* count (``n_in - round(n_in*(1-sparsity))``) rather
    than flooring ``(1-sparsity)*n_in`` directly -- the latter compounds
    floating-point error (e.g. 12*(1 - 5/12) landing on 6.999999999999999
    instead of 7) into an off-by-one keep-count, exactly the bug found
    and fixed in `obs_projection.py`'s block-level rounding.
    """
    return max(1, n_in - round(n_in * sparsity))


def _sample_batch(calib_ids: torch.Tensor, n: int, device) -> torch.Tensor:
    idx = torch.randint(0, calib_ids.shape[0], (n,))
    return calib_ids[idx].to(device)


def _set_trainable(modules: list[nn.Linear], flag: bool) -> None:
    for m in modules:
        m.weight.requires_grad_(flag)
        if m.bias is not None:
            m.bias.requires_grad_(flag)


def _build_optimizer(
    trainable_modules: list[nn.Linear], lr: float, betas: tuple[float, float] = (0.9, 0.999),
    extra_params: list[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    """One Adam instance over every trainable module's weight (+ bias).

    `foreach=False`: PyTorch's default multi-tensor ("foreach") Adam path
    fuses the update across every passed-in tensor via ops like
    `torch._foreach_sqrt`, which needs a temporary buffer sized to *all*
    of them at once -- a real extra peak-memory spike on top of the
    already-resident optimizer state, not just a speed optimization.
    Measured: this alone OOM'd Phase 2 on HGRN-1.3B (168 simultaneously
    trainable layers) even after every other memory fix in this file.
    `foreach=False` falls back to a per-parameter loop -- identical Adam
    math, no numerical difference, just lower peak memory at some speed
    cost. Phase 1 only ever passes one module at a time, so this doesn't
    change anything there.

    `betas` defaults to PyTorch's stock (0.9, 0.999) -- NOT what real ELSA
    actually configures (0.9, 0.95) via its admm_beta1/admm_beta2 training
    args (ELSA-official/lib/trainer.py). Pass `betas=(0.9, 0.95)` (see
    Phase1Config/Phase2Config's beta1/beta2 fields, wired from
    run_pipeline.py's --adam-beta1/--adam-beta2) to match real ELSA.
    """
    params = []
    for m in trainable_modules:
        params.append(m.weight)
        if m.bias is not None:
            params.append(m.bias)
    params += list(extra_params or [])
    return torch.optim.Adam(params, lr=lr, betas=betas, foreach=False)


def _build_linear_decay_scheduler(opt: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear decay from ``lr`` at step 0 to 0 at ``total_steps``, matching
    real ELSA's Table 4 ("LR schedule: Linear decay"). One ``.step()`` per
    optimizer step, not per round -- Phase 2 only (see `_x_step`'s
    docstring for why Phase 1 doesn't get this)."""
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps))


def _cosine_lambda_ramp(step: int, total_steps: int, lambda_max: float) -> float:
    """Cosine ramp from 0 up to lambda_max over total_steps.

    Early in a run the penalty is near zero, so x trains almost freely
    against the true loss; as it ramps up, x gets pulled toward the
    repeatedly-reprojected sparse target -- a smooth transition instead
    of clamping straight to the sparsity constraint from step 1. Used by
    both Phase 1 (per layer) and Phase 2 (globally). Previously Phase 2
    used an ad hoc linear-in-round formula mislabeled "cosine-ish"; this
    is the real cosine ramp EHWS/EHWS-U both use, now shared by both
    phases here too.
    """
    if total_steps <= 0:
        return lambda_max
    progress = min(1.0, step / total_steps)
    return lambda_max * 0.5 * (1.0 - math.cos(math.pi * progress))


def _x_step(
    model: nn.Module,
    teacher: nn.Module,
    trainable_modules: list[nn.Linear],
    prox_targets: list[tuple[nn.Linear, torch.Tensor, torch.Tensor]],
    calib_ids: torch.Tensor,
    n_steps: int,
    micro_batch: int,
    grad_accum: int,
    lr: float,
    admm_lambda: float,
    alpha_kd: float,
    device,
    max_grad_norm: float = 1.0,
    opt: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    kd_temperature: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    prox_after_clip: bool = True,
    extra_params: list[torch.nn.Parameter] | None = None,
    sampler=None,
) -> torch.optim.Optimizer:
    """Run n_steps of Adam minimizing f(x) + (lambda/2)||x-target||_F^2.

    Gradients are clipped to ``max_grad_norm`` before each optimizer step,
    matching real ELSA's ``clip_grad_norm_`` call in its trainer.

    ``prox_after_clip=True`` (default since 2026-09-11): matches real
    ELSA's actual ``ADMMOptimizer._proximal_update``
    (ELSA-official/lib/optimizers.py) exactly -- only the task loss is
    backpropped and clipped, then the proximal gradient
    ``admm_lambda * (w - target)`` (algebraically identical to ELSA's own
    ``lmda * (w - split + dual)``, since `target` here is already `z - u`)
    is added to `.grad` *after* clipping, so the ADMM constraint pull is
    never attenuated by the clip -- ELSA's own comment: "This ensures
    proximal is not clipped." Beat the prior default on both WT2 (85.98
    vs 90.59) and C4 (52.68 vs 53.86) @80% sparsity on OPT-125M (see
    PROGRESS.md, opt-125m-fixablation-prox-after-clip). ``prox_after_clip
    =False`` (original EHWS behavior): the proximal term is added
    directly into the backpropped loss, so ``clip_grad_norm_`` clips the
    *combined* task+proximal gradient together.

    ``opt``, if given, is reused as-is (its momentum/second-moment state
    carries over) instead of building a fresh Adam instance. Phase 2
    passes one optimizer built once for the whole run -- matching real
    ELSA's continuous-training design (one persistent optimizer, dual
    updates only every `admm_interval` steps) instead of restarting Adam
    from scratch every ADMM round, which was forcing repeated
    re-warm-up and was a real contributor to the divergence diagnosed
    this session. Phase 1 still builds a fresh optimizer per call
    (``opt=None``, unchanged) since each layer's own run is a logically
    separate sub-problem.

    ``scheduler``, if given, is stepped once per optimizer step (real
    ELSA's Table 4 "LR schedule: Linear decay" -- see
    `_build_linear_decay_scheduler`). Phase 1 doesn't pass one: it has no
    real-ELSA analogue to match, and its optimizer is rebuilt fresh every
    call anyway (see above), so there's no single continuous run for a
    schedule to decay across.

    ``prox_targets`` is a list of ``(module, z, u)``; the penalty pulls the
    module's weight toward ``z - u``. The target is formed per module, in
    float32, when it's used rather than stored, so Phase 2 doesn't keep an
    extra full copy of every weight alive.

    ``extra_params`` (Phase 2 ``train_extra``) are trained by the same
    optimizer and clipped with everything else, with no penalty. ``sampler``
    (``EpochSampler``) replaces sampling with replacement.
    """
    for p, _, _ in prox_targets:
        p.weight.requires_grad_(True)
        if p.bias is not None:
            p.bias.requires_grad_(True)
    params = []
    for m in trainable_modules:
        params.append(m.weight)
        if m.bias is not None:
            params.append(m.bias)
    params += list(extra_params or [])
    if opt is None:
        opt = _build_optimizer(trainable_modules, lr, betas=betas)

    def target(z, u, like):
        return (z.float() - u.float()).to(like.dtype)

    model.train()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = sampler(micro_batch, device) if sampler is not None else _sample_batch(calib_ids, micro_batch, device)
            with _student_autocast(device):
                student_logits = model(input_ids=batch, use_cache=False).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch, use_cache=False).logits
            loss, _ = combined_loss(student_logits, teacher_logits, batch, alpha_kd, kd_temperature)

            if prox_after_clip:
                # Only the task loss is backpropped/clipped here -- the
                # proximal gradient is added below, after clip_grad_norm_.
                (loss / grad_accum).backward()
            else:
                prox = student_logits.new_zeros(())
                for module, z, u in prox_targets:
                    prox = prox + 0.5 * admm_lambda * torch.sum((module.weight - target(z, u, module.weight)) ** 2)
                (loss / grad_accum + prox / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if prox_after_clip:
            with torch.no_grad():
                for module, z, u in prox_targets:
                    module.weight.grad.add_(admm_lambda * (module.weight.detach() - target(z, u, module.weight)))
        opt.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return opt


class _StopForward(Exception):
    """Raised from a Hessian hook once every hooked layer has seen its input,
    so the rest of the forward pass (later blocks, lm_head) is skipped."""


@torch.no_grad()
def _compute_hessians(
    model: nn.Module,
    modules: list[nn.Linear],
    calib_ids: torch.Tensor,
    device,
    damping: float,
    n_samples: int | None = None,
    diag_only: bool = False,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Damped Hessians for every module in ``modules`` from one calibration pass.

    Three speedups over measuring each layer with its own full-model pass:

    - All of ``modules`` are hooked at once, so N layers cost one pass
      instead of N. Only valid when none of them changes weight between
      measurements -- true for `build_dense_states` (nothing is pruned
      there), not for `run_phase1` (each layer is pruned before the next
      one is measured), which still passes one module at a time.
    - Each batch's forward pass stops as soon as every hooked module has
      seen its input (`_StopForward`), skipping every later block and the
      lm_head. Nothing downstream of the last hooked layer can affect its
      input, so this is exact.
    - ``n_samples`` measures on only the first ``n_samples`` calibration
      sequences (``None`` = all of them, the original behavior). This one
      is an approximation; a diagonal converges quickly, so a few hundred
      2048-token sequences is plenty.

    ``diag_only`` returns ``(d_in,)`` diagonals instead of full matrices
    (see `LayerHessian`). ``dtype`` defaults to each module's weight dtype.
    """
    hessians = [LayerHessian(m.weight.shape[1], device, diag_only=diag_only) for m in modules]
    pending: set[int] = set()

    def make_hook(i: int, hess: LayerHessian):
        def hook(module, inputs):
            hess.update(inputs[0])
            pending.discard(i)
            if not pending:
                raise _StopForward
        return hook

    handles = [m.register_forward_pre_hook(make_hook(i, h)) for i, (m, h) in enumerate(zip(modules, hessians))]
    model.eval()
    ids = calib_ids if n_samples is None else calib_ids[:n_samples]
    batch_size = 4
    try:
        for i in range(0, ids.shape[0], batch_size):
            pending.update(range(len(modules)))
            batch = ids[i : i + batch_size].to(device)
            try:
                with _student_autocast(device):
                    model(input_ids=batch, use_cache=False)
            except _StopForward:
                pass
    finally:
        for h in handles:
            h.remove()
    return [h.damped(damping=damping, dtype=dtype or m.weight.dtype) for m, h in zip(modules, hessians)]


def _compute_hessian(
    model: nn.Module, module: nn.Linear, calib_ids: torch.Tensor, seqlen: int, device, damping: float,
    n_samples: int | None = None, diag_only: bool = False,
) -> torch.Tensor:
    return _compute_hessians(model, [module], calib_ids, device, damping, n_samples=n_samples, diag_only=diag_only)[0]


def build_dense_states(
    model: nn.Module,
    layers: list[PrunableLayer],
    calib_ids: torch.Tensor,
    seqlen: int,
    device,
    damping: float,
    log_fn=print,
    hessian_n_calib: int | None = None,
    diag_only: bool = False,
) -> dict[str, LayerState]:
    """Cold-start ablation: build Phase 2's warm-start dict directly from
    the dense model (x=z=x0, u=0), skipping Phase 1 entirely.

    Isolates whether Phase 1's per-layer warm-start is actually helping
    Phase 2, or whether Phase 2 alone (real ELSA's own design -- one
    continuous ADMM trajectory from the dense model, no per-layer
    pre-pass) does just as well or better. Mirrors
    `Extreme_Layer_Global_Pruning_Unstructured`'s own `build_dense_states`
    ablation, same rationale, adapted for this package's uniform-ratio
    Phase 2.
    """
    # No weight changes here, so every layer's Hessian comes from the same
    # dense model and they can all share calibration passes (see
    # _compute_hessians). Full matrices are accumulated in float64, so
    # layers are grouped to cap that at ~4GB per pass; diagonals all fit in one.
    budget = 4 * 1024**3
    groups: list[list[PrunableLayer]] = [[]]
    used = 0
    for pl in layers:
        need = 8 * pl.module.weight.shape[1] * (1 if diag_only else pl.module.weight.shape[1])
        if groups[-1] and used + need > budget:
            groups.append([])
            used = 0
        groups[-1].append(pl)
        used += need

    states: dict[str, LayerState] = {}
    done = 0
    for group in groups:
        Hs = _compute_hessians(
            model, [pl.module for pl in group], calib_ids, device, damping,
            n_samples=hessian_n_calib, diag_only=diag_only,
        )
        for pl, H in zip(group, Hs):
            module = pl.module
            n_out, n_in = module.weight.shape
            x0 = module.weight.data.clone()
            states[pl.name] = LayerState(
                name=pl.name, module=module, x=x0.clone(), z=x0.clone(), u=torch.zeros_like(x0), H=H.cpu(),
                n_in=n_in, n_out=n_out, x0_norm=x0.norm().item(), final_sparsity=0.0,
            )
        done += len(group)
        log_fn(f"[cold-start] Hessians computed for {done}/{len(layers)} layers, dense (no Phase 1)")
    return states


def run_phase1(
    model: nn.Module,
    teacher: nn.Module,
    layers: list[PrunableLayer],
    calib_ids: torch.Tensor,
    cfg: Phase1Config,
    device,
    log_fn=print,
    zstep_fn=diagonal_project,
) -> dict[str, LayerState]:
    """Sequential per-layer ADMM: one continuous run per layer straight at
    cfg.target_sparsity, penalty cosine-ramped 0 -> admm_lambda_max.

    No saturation ladder -- see this module's docstring for why the
    ladder that used to be here was removed. cfg.target_sparsity is a
    fixed reference point every layer warm-starts toward, independent of
    whatever overall sparsity Phase 2 is eventually asked for -- Phase
    2's own Z-step re-derives its keep-count against the real target
    every round regardless of Phase 1's exact per-layer support, so a
    lower final target simply lets some of these coordinates back in
    during Phase 2. That's what lets Phase 1 run once per model and get
    reused, warm-started, across every Phase 2 sparsity level.
    """
    _set_trainable([pl.module for pl in layers], False)
    states: dict[str, LayerState] = {}
    total_steps = cfg.rounds * cfg.x_steps

    for li, pl in enumerate(layers):
        module = pl.module
        n_out, n_in = module.weight.shape
        x0 = module.weight.data.clone()
        x0_norm = x0.norm().item()

        H = _compute_hessian(
            model, module, calib_ids, cfg.seqlen, device, cfg.damping,
            n_samples=cfg.hessian_n_calib, diag_only=cfg.diag_only_hessian,
        )
        k_keep = _k_keep(n_in, cfg.target_sparsity)

        x = x0.clone()
        z = x0.clone()
        u = torch.zeros_like(x0)

        # One optimizer for this layer's whole run, reused across rounds
        # (see _x_step's docstring). Still rebuilt fresh per layer --
        # layers are genuinely separate sub-problems.
        opt = _build_optimizer([module], cfg.lr, betas=(cfg.beta1, cfg.beta2))

        for r in range(cfg.rounds):
            if cfg.lambda_schedule == "constant":
                admm_lambda = cfg.admm_lambda_max
            else:
                admm_lambda = _cosine_lambda_ramp((r + 1) * cfg.x_steps, total_steps, cfg.admm_lambda_max)
            module.weight.data.copy_(x)
            _x_step(
                model, teacher, [module], [(module, z, u)],
                calib_ids, cfg.x_steps, cfg.micro_batch, cfg.grad_accum,
                cfg.lr, admm_lambda, cfg.alpha_kd, device,
                max_grad_norm=cfg.max_grad_norm, opt=opt, kd_temperature=cfg.kd_temperature,
                betas=(cfg.beta1, cfg.beta2), prox_after_clip=cfg.prox_after_clip,
            )
            x = module.weight.data.clone()

            v = x + u
            z = zstep_fn(v, H, k_keep)
            u = u + x - z

        module.weight.data.copy_(z)
        _set_trainable([module], False)
        actual_sparsity = (z == 0).float().mean().item()
        # Offload this layer's Hessian to CPU now that its own round loop
        # (the only place that needs it on-device) is done -- otherwise
        # every finished layer's full (d_in x d_in) Hessian stays resident
        # on GPU for the rest of Phase 1, which OOMs on models with many
        # wide prunable layers (measured: ~11GB for HGRN-1.3B's 168
        # layers, even after halving to float32 in hessian.py). Moved
        # back to GPU in bulk at the start of run_phase2, which is the
        # next (and only other) place any of this is read.
        states[pl.name] = LayerState(
            name=pl.name, module=module, x=x, z=z, u=u, H=H.cpu(),
            n_in=n_in, n_out=n_out, x0_norm=x0_norm,
            final_sparsity=actual_sparsity, admm_lambda=admm_lambda,
        )
        log_fn(
            f"[phase1] layer {li+1}/{len(layers)} {pl.name}: DONE, "
            f"target_sparsity={cfg.target_sparsity:.2f} final_sparsity={actual_sparsity:.3f}"
        )

    return states


# --------------------------------------------------------------------------
# Local Phase 1 (`--p1-mode local`): per-layer reconstruction ADMM
# --------------------------------------------------------------------------
#
# `run_phase1` above optimizes each layer against the *full-model* CE+KD loss:
# every x-step runs the whole student forward/backward plus a teacher forward,
# 56 of them per layer. That's 72*56 = 4032 full-model steps on OPT-125M and
# 168*56 = 9408 on HGRN-1.3B -- as much as, or more than, Phase 2's entire
# 4096-step budget, which is why two-phase has been slower than one-phase
# (PROGRESS.md Session 12).
#
# Local mode instead solves each layer's own reconstruction problem,
#     min_W  1/2 tr((W - W0) H (W - W0)^T)   s.t. W semi-structured sparse,
# H = (2/N) X X^T of the layer's inputs (the Hessian this package already
# measures). That's ||(W - W0) X||^2 / N: keep the layer's output close to the
# dense layer's on the same inputs. Its x-step has a closed form, so no
# backprop and no teacher: the only model forward passes are the calibration
# passes that measure H. ADMM with an increasing penalty (the ALPS recipe),
# then an exact least-squares refit of the kept weights on the final support.


@dataclasses.dataclass
class LocalPhase1Config:
    target_sparsity: float = 0.8
    rounds: int = 50
    # Penalty rho, relative to mean(diag(H)), grown geometrically from
    # rho_start to rho_end over `rounds` so x is pulled onto z by the end.
    rho_start: float = 0.01
    rho_end: float = 10.0
    # Least-squares refit of the surviving weights on the final support.
    refit: bool = True
    damping: float = 0.01
    hessian_n_calib: int | None = None
    # Hand Phase 2 only diag(H) (what the diagonal/magnitude Z-steps read).
    diag_only_hessian: bool = True


def _recon_error(W: torch.Tensor, W0: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W-W0) H (W-W0)^T) / tr(W0 H W0^T): relative output error on the calibration inputs."""
    D = W - W0
    return ((D @ H) * D).sum().item() / max(((W0 @ H) * W0).sum().item(), 1e-12)


def _refit_on_support(W0: torch.Tensor, H: torch.Tensor, B: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Exact per-row least squares on a fixed support.

    Row i keeps coordinates idx[i] (k of them) and solves
    min_w 1/2 (w - w0)^T H (w - w0) with w zero off the support, i.e.
    H_SS w_S = (H w0)_S = B[i, S] (B = W0 H, H symmetric). Every row keeps
    exactly k weights (semi-structured), so rows batch into one (R, k, k)
    Cholesky solve; chunked to bound the gathered H_SS memory.
    """
    d_out, k = idx.shape
    out = torch.zeros_like(W0)
    chunk = max(1, int(2**28 // (k * k)))  # ~1GB of float32 H_SS per chunk
    for i in range(0, d_out, chunk):
        rows = idx[i : i + chunk]
        Hss = H[rows[:, :, None], rows[:, None, :]]
        rhs = B[i : i + chunk].gather(1, rows).unsqueeze(-1)
        L = torch.linalg.cholesky(Hss)
        out[i : i + chunk].scatter_(1, rows, torch.cholesky_solve(rhs, L).squeeze(-1))
    return out


@torch.no_grad()
def _local_admm_layer(
    W0: torch.Tensor, H: torch.Tensor, k_keep: int, cfg: LocalPhase1Config, zstep_fn,
) -> tuple[torch.Tensor, dict]:
    """ADMM on one layer's reconstruction objective; returns (pruned W, stats)."""
    W0 = W0.float()
    H = H.float()
    evals, Q = torch.linalg.eigh(H)  # H + rho I = Q diag(evals + rho) Q^T for every rho
    evals = evals.clamp_min(0.0)
    B = W0 @ H
    scale = torch.diagonal(H).mean().item()

    z0 = zstep_fn(W0, H, k_keep)  # one-shot projection of the dense weights, for comparison
    z, u = z0.clone(), torch.zeros_like(W0)
    rho_prev = None
    for r in range(cfg.rounds):
        frac = r / max(cfg.rounds - 1, 1)
        rho = scale * cfg.rho_start * (cfg.rho_end / cfg.rho_start) ** frac
        if rho_prev is not None:
            u.mul_(rho_prev / rho)  # scaled dual: keep rho*u fixed when rho changes
        rho_prev = rho
        # x-step: argmin 1/2 tr((W-W0)H(W-W0)^T) + rho/2 ||W - (z-u)||^2
        #       = (W0 H + rho (z - u)) (H + rho I)^-1
        x = (((B + rho * (z - u)) @ Q) / (evals + rho)) @ Q.t()
        z = zstep_fn(x + u, H, k_keep)
        u.add_(x - z)

    stats = {"err_oneshot": _recon_error(z0, W0, H), "err_admm": _recon_error(z, W0, H)}
    if cfg.refit:
        idx = z.abs().topk(k_keep, dim=1).indices
        z = _refit_on_support(W0, H, B, idx)
        stats["err_refit"] = _recon_error(z, W0, H)
    return z, stats


@torch.no_grad()
def _shared_input_groups(model: nn.Module, layers: list[PrunableLayer], sample: torch.Tensor) -> list[list[PrunableLayer]]:
    """Group consecutive layers that read the exact same input tensor (e.g. q/k/v).

    Pruning one member can't change another's input, so a group's Hessians
    can come from one calibration pass and its layers can be pruned together
    -- the same result as one-at-a-time, with fewer passes. Detected by input
    data_ptr on one short forward; every captured input is kept alive until
    the pass ends so a freed buffer can't be reused at the same address.
    """
    seen: dict[int, tuple[int, tuple]] = {}
    keep_alive = []

    def make_hook(i):
        def hook(module, inputs):
            x = inputs[0]
            keep_alive.append(x)
            seen[i] = (x.data_ptr(), tuple(x.shape))
        return hook

    handles = [pl.module.register_forward_pre_hook(make_hook(i)) for i, pl in enumerate(layers)]
    try:
        model(input_ids=sample, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    keep_alive.clear()

    groups: list[list[PrunableLayer]] = []
    prev = None
    for i, pl in enumerate(layers):
        key = seen.get(i)
        if groups and key is not None and key == prev:
            groups[-1].append(pl)
        else:
            groups.append([pl])
        prev = key
    return groups


def run_phase1_local(
    model: nn.Module,
    layers: list[PrunableLayer],
    calib_ids: torch.Tensor,
    cfg: LocalPhase1Config,
    device,
    log_fn=print,
    zstep_fn=diagonal_project,
) -> dict[str, LayerState]:
    """Sequential layer-wise reconstruction pruning, straight to cfg.target_sparsity.

    Layers go in execution order; each group of shared-input layers is
    measured on the model with every earlier layer already pruned (as in
    `run_phase1`), then committed to its pruned weights. Biases are left
    untouched.

    Handoff to Phase 2: x = z (the pruned weights), u = 0, admm_lambda = 0.
    Phase 2 starts from the actually-pruned model with a fresh dual -- this
    phase's dual belongs to a different objective (reconstruction error, not
    CE+KD) and a different penalty scale, so there is nothing meaningful to
    carry or rescale.
    """
    import time

    t_start = time.time()
    groups = _shared_input_groups(model, layers, calib_ids[:1, :128].to(device))
    log_fn(f"[phase1-local] {len(layers)} layers in {len(groups)} shared-input groups")
    states: dict[str, LayerState] = {}
    done = 0
    for group in groups:
        t0 = time.time()
        Hs = _compute_hessians(
            model, [pl.module for pl in group], calib_ids, device, cfg.damping,
            n_samples=cfg.hessian_n_calib, diag_only=False, dtype=torch.float32,  # eigh/Cholesky need more than bf16
        )
        t_h = time.time() - t0
        for pl, H in zip(group, Hs):
            t1 = time.time()
            module = pl.module
            n_out, n_in = module.weight.shape
            x0 = module.weight.data.clone()
            k_keep = _k_keep(n_in, cfg.target_sparsity)
            z, stats = _local_admm_layer(x0, H, k_keep, cfg, zstep_fn)
            z = z.to(x0.dtype)
            module.weight.data.copy_(z)
            done += 1
            H_keep = (torch.diagonal(H).clone() if cfg.diag_only_hessian else H).to(x0.dtype)
            # Handoff tensors live on CPU (the live module already holds z on the
            # GPU; run_phase2 moves what it needs back). x and z are the same tensor.
            z_cpu = z.cpu()
            states[pl.name] = LayerState(
                name=pl.name, module=module, x=z_cpu, z=z_cpu, u=torch.zeros_like(z_cpu), H=H_keep.cpu(),
                n_in=n_in, n_out=n_out, x0_norm=x0.norm().item(),
                final_sparsity=(z == 0).float().mean().item(), admm_lambda=0.0,
            )
            errs = " ".join(f"{k}={v:.4f}" for k, v in stats.items())
            log_fn(
                f"[phase1-local] layer {done}/{len(layers)} {pl.name}: {errs} "
                f"sparsity={states[pl.name].final_sparsity:.3f} (hessian {t_h:.1f}s, solve {time.time()-t1:.1f}s)"
            )
        del Hs
    log_fn(f"[phase1-local] done in {time.time()-t_start:.1f}s")
    return states


def run_phase2(
    model: nn.Module,
    teacher: nn.Module,
    phase1_states: dict[str, LayerState],
    calib_ids: torch.Tensor,
    cfg: Phase2Config,
    device,
    log_fn=print,
    zstep_fn=diagonal_project,
) -> dict[str, LayerState]:
    """Joint global ADMM, uniform per-layer target sparsity `cfg.target_sparsity`.

    ``zstep_fn`` defaults to ``diagonal_project`` -- real ELSA's own
    Z-step (Diag(H) projection, verified against the official ELSA repo;
    see `ehws/diagonal_projection.py`), per the user's explicit choice
    among the three candidate readings of slide 39's Z-step (real ELSA's
    Diag(H) vs. CWS's slide-30 incremental scheme vs. an exact
    full-Hessian solver). ``obs_project`` (`ehws/obs_projection.py`, the
    exact full-Hessian-aware block solver previously used here) remains
    available as the injectable alternative for controlled ablations.
    """
    working: dict[str, LayerState] = {}
    modules = []
    if cfg.dual_init not in ("carry", "rescale", "zero"):
        raise ValueError(f"unknown dual_init {cfg.dual_init!r}")
    for name, s in phase1_states.items():
        sd = cfg.state_dtype or s.module.weight.dtype
        z = s.z.to(device=device, dtype=sd, copy=True)
        u = s.u.to(device=device, dtype=sd, copy=True)
        if cfg.dual_init == "zero":
            u.zero_()
        elif cfg.dual_init == "rescale" and s.admm_lambda > 0:
            u.mul_(s.admm_lambda / cfg.admm_lambda_max)
        s.module.weight.data.copy_(s.x)
        # Phase 1 (or build_dense_states) offloaded each H to CPU once it
        # was done with it; Phase 2 needs every layer's H simultaneously
        # and repeatedly (every round's Z-step), so bring them all back
        # now, once, rather than per-round. x isn't kept here: the live
        # module weight *is* x for the whole run.
        working[name] = LayerState(
            name=name, module=s.module, x=None, z=z, u=u, H=s.H.to(device),
            n_in=s.n_in, n_out=s.n_out, x0_norm=s.x0_norm,
        )
        modules.append(s.module)

    _set_trainable(modules, True)
    k_keep = {name: _k_keep(s.n_in, cfg.target_sparsity) for name, s in working.items()}
    if cfg.hessian_refresh and any(s.H.dim() != 1 for s in working.values()):
        raise ValueError("hessian_refresh only re-measures diag(H); use it with --zstep diagonal (diag-only Hessians)")

    extra_params: list[torch.nn.Parameter] = []
    if cfg.train_extra:
        owned = {id(p) for m in modules for p in m.parameters()}
        if cfg.train_extra_scope == "norms":
            pool = [p for m in model.modules() if "norm" in type(m).__name__.lower() for p in m.parameters(recurse=False)]
        elif cfg.train_extra_scope == "all":
            pool = list(model.parameters())
        else:
            raise ValueError(f"unknown train_extra_scope {cfg.train_extra_scope!r}")
        seen: set[int] = set()
        extra_params = [p for p in pool if id(p) not in owned and not (id(p) in seen or seen.add(id(p)))]
        for p in extra_params:
            p.requires_grad_(True)
        log_fn(f"[phase2] also training {len(extra_params)} non-pruned tensors "
               f"({sum(p.numel() for p in extra_params)/1e6:.1f}M params: norms, embeddings, lm_head, ...)")

    # Hessian refresh: accumulate sum(x^2) per input column of every prunable
    # layer over the x-step's student forwards. With gradient checkpointing
    # the recompute pass fires the hook again; sum and count both double, so
    # the average is unchanged.
    hooks = []
    if cfg.hessian_refresh:
        acc = {name: [torch.zeros(s.n_in, device=device, dtype=torch.float32), 0] for name, s in working.items()}

        def make_hook(name):
            def hook(module, inputs):
                if not module.training:
                    return
                with torch.no_grad():
                    a = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                    acc[name][0].add_(a.float().pow(2).sum(dim=0))
                    acc[name][1] += a.shape[0]
            return hook

        hooks = [s.module.register_forward_pre_hook(make_hook(name)) for name, s in working.items()]

    # One optimizer for the whole Phase 2 run (real ELSA's continuous-training
    # design: persistent Adam state, dual updates only every `rounds`-th
    # boundary -- not a fresh Adam restarted at every round, which was
    # forcing repeated momentum re-warm-up and was diagnosed as a real
    # contributor to divergence this session).
    opt = _build_optimizer(modules, cfg.lr, betas=(cfg.beta1, cfg.beta2), extra_params=extra_params)
    total_steps = cfg.rounds * cfg.x_steps
    scheduler = _build_linear_decay_scheduler(opt, total_steps=total_steps)
    sampler = EpochSampler(calib_ids, seed=cfg.seed) if cfg.epoch_sampling else None

    try:
        for r in range(cfg.rounds):
            if cfg.lambda_schedule == "constant":
                admm_lambda = cfg.admm_lambda_max
            else:
                admm_lambda = _cosine_lambda_ramp((r + 1) * cfg.x_steps, total_steps, cfg.admm_lambda_max)
            prox_targets = [(s.module, s.z, s.u) for s in working.values()]
            _x_step(
                model, teacher, modules, prox_targets, calib_ids,
                cfg.x_steps, cfg.micro_batch, cfg.grad_accum, cfg.lr,
                admm_lambda, cfg.alpha_kd, device,
                max_grad_norm=cfg.max_grad_norm, opt=opt, scheduler=scheduler, kd_temperature=cfg.kd_temperature,
                betas=(cfg.beta1, cfg.beta2), prox_after_clip=cfg.prox_after_clip,
                extra_params=extra_params, sampler=sampler,
            )

            last = r == cfg.rounds - 1
            max_ratio, res_sq, x_sq, h_drift = 0.0, 0.0, 0.0, 0.0
            for name, s in working.items():
                if cfg.hessian_refresh and acc[name][1] > 0:
                    h = acc[name][0] * (2.0 / acc[name][1])
                    h[h == 0] = 1.0
                    h += cfg.damping * h.mean()
                    old = s.H.float()
                    if cfg.hessian_ema > 0:
                        h = cfg.hessian_ema * old + (1.0 - cfg.hessian_ema) * h
                    h_drift = max(h_drift, ((h / h.mean()) - (old / old.mean())).abs().mean().item())
                    s.H = h.to(s.H.dtype)
                    acc[name][0].zero_()
                    acc[name][1] = 0
                x = s.module.weight.data.float()
                u = s.u.float()
                z = zstep_fn(x + u, s.H, k_keep[name])
                u = u + x - z
                res_sq += (x - z).pow(2).sum().item()
                x_sq += x.pow(2).sum().item()
                s.z, s.u = z.to(s.z.dtype), u.to(s.u.dtype)
                if last:
                    # Final commit straight from the float32 projection (not
                    # the possibly-bf16 stored z).
                    s.module.weight.data.copy_(z)
                max_ratio = max(max_ratio, u.norm().item() / max(s.x0_norm, 1e-12))
                del x, u, z
            msg = (f"[phase2] round {r+1}/{cfg.rounds} target_sparsity={cfg.target_sparsity:.2f} "
                   f"max ||u||/||x0||={max_ratio:.4f} ||x-z||/||x||={math.sqrt(res_sq / max(x_sq, 1e-30)):.4f}")
            if cfg.hessian_refresh:
                msg += f" diagH_drift={h_drift:.4f}"
            log_fn(msg)
    finally:
        for h in hooks:
            h.remove()

    for name, s in working.items():
        s.final_sparsity = (s.module.weight.data == 0).float().mean().item()
    _set_trainable(modules, False)
    for p in extra_params:
        p.requires_grad_(False)
    # Free gradients (and with them ~5GB at fp32 on HGRN) before evaluation.
    for p in model.parameters():
        p.grad = None
    return working

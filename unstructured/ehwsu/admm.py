"""Phase 1 (layer-wise ADMM) and Phase 2 (global ADMM, unstructured) drivers.

This is the unstructured sibling of the semi-structured `ehws/admm.py`:
Phase 2 here spends one shared, global weight budget
(`k_keep_total = round((1-s) * N_total)`) across the whole network at
once, via `global_projection.global_diagonal_project`, instead of giving
every layer the same fixed sparsity ratio. Layers that turn out to be
more redundant give up proportionally more of that budget; layers that
matter more keep proportionally more -- nothing forces any two layers to
land at the same sparsity, which is what makes the result genuinely
unstructured rather than layer-uniform.

Phase 1 -- one continuous ADMM run per layer, straight at
`cfg.target_sparsity`, with the penalty ramped smoothly instead of a
discrete ladder:

    x-step:  x_l <- argmin_x  f(x_l) + (lambda_t/2) ||x_l - z_l + u_l||_F^2
    z-step:  z_l <- argmin_{||z_l||_0<=k_l} (z_l - v_l)^T H_l (z_l - v_l)
    u-step:  u_l <- u_l + x_l^{t+1} - z_l^{t+1}
    lambda_t: cosine ramp 0 -> admm_lambda_max over the run.

Phase 2 -- same X/U mechanics, global Z-step:

    x-step: all layers' x_l trained jointly under one global forward pass
    z-step: argmin_{sum_l ||z_l||_0 <= k_total} sum_l (z_l-v_l)^T H_l (z_l-v_l)
            (one shared count budget, no per-layer split)
    u-step: per layer, same as Phase 1.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from .diagonal_projection import diagonal_project
from .global_projection import global_diagonal_project
from .hessian import LayerHessian
from .losses import combined_loss
from .model_layers import PrunableLayer


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


@dataclasses.dataclass
class Phase1Config:
    # Fixed per-layer warm-start target, independent of whatever overall
    # sparsity Phase 2 is eventually asked for. Phase 2 resumes from
    # Phase 1's trained x (not its hard-zeroed z) and re-derives its own
    # k_keep from scratch every round regardless of Phase 1's exact
    # per-layer support, so this doesn't need to match Phase 2's target --
    # that's what lets Phase 1 run once per model and get reused across
    # every Phase 2 sparsity level. 0.7 is from an empirical sweep over
    # {0.3, 0.5, 0.7, 0.9} against a Phase 2 target of 0.5 -- see README.
    target_sparsity: float = 0.7
    # rounds * x_steps = 56 total optimizer steps per layer.
    rounds: int = 14
    x_steps: int = 4
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    admm_lambda_max: float = 5e-5
    alpha_kd: float = 0.5
    damping: float = 0.01
    seqlen: int = 2048
    max_grad_norm: float = 1.0


@dataclasses.dataclass
class Phase2Config:
    target_sparsity: float = 0.5  # overall fraction of all prunable weights pruned network-wide, not a per-layer ratio
    # rounds * x_steps = 4096 total optimizer steps, with the ADMM
    # dual/z update every 32 steps.
    rounds: int = 128
    x_steps: int = 32
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 1e-4
    admm_lambda_max: float = 5e-5
    lambda_schedule: str = "cosine"  # "cosine": ramp 0->max over the run. "constant": admm_lambda_max from round 1.
    alpha_kd: float = 0.5
    seqlen: int = 2048
    max_grad_norm: float = 1.0


def _k_keep(n_in: int, sparsity: float) -> int:
    """Nonzeros to keep per row for a given fraction pruned.

    Rounds the *pruned* count (`n_in - round(n_in * (1 - sparsity))`)
    rather than flooring `(1 - sparsity) * n_in` directly, which can land
    one entry short from ordinary floating-point error.
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


def _build_optimizer(trainable_modules: list[nn.Linear], lr: float) -> torch.optim.Optimizer:
    """One Adam instance over every trainable module's weight (+ bias)."""
    params = []
    for m in trainable_modules:
        params.append(m.weight)
        if m.bias is not None:
            params.append(m.bias)
    return torch.optim.Adam(params, lr=lr)


def _build_linear_decay_scheduler(opt: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear decay from lr at step 0 to 0 at total_steps. Phase 2 only."""
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps))


def _cosine_lambda_ramp(step: int, total_steps: int, lambda_max: float) -> float:
    """Cosine ramp from 0 up to lambda_max over total_steps.

    Early in a run the penalty is near zero, so x trains almost freely
    against the true loss; as it ramps up, x gets pulled toward the
    repeatedly-reprojected sparse target -- a smooth transition instead
    of clamping straight to the sparsity constraint from step 1. Used by
    both Phase 1 (per layer) and Phase 2 (globally).
    """
    if total_steps <= 0:
        return lambda_max
    progress = min(1.0, step / total_steps)
    return lambda_max * 0.5 * (1.0 - math.cos(math.pi * progress))


def _x_step(
    model: nn.Module,
    teacher: nn.Module,
    trainable_modules: list[nn.Linear],
    prox_targets: list[tuple[nn.Linear, torch.Tensor]],
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
) -> torch.optim.Optimizer:
    """Run n_steps of Adam minimizing f(x) + (lambda/2) * ||x - target||_F^2.

    `opt`, if given, is reused as-is so its momentum carries across
    calls: Phase 2 builds one optimizer for its whole run, and Phase 1
    builds one per layer and reuses it across that layer's rounds --
    rebuilding Adam from scratch on every round forces the momentum to
    re-warm-up every time and measurably hurts convergence.
    """
    for p, _ in prox_targets:
        p.weight.requires_grad_(True)
        if p.bias is not None:
            p.bias.requires_grad_(True)
    params = []
    for m in trainable_modules:
        params.append(m.weight)
        if m.bias is not None:
            params.append(m.bias)
    if opt is None:
        opt = _build_optimizer(trainable_modules, lr)

    model.train()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = _sample_batch(calib_ids, micro_batch, device)
            student_logits = model(input_ids=batch, use_cache=False).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch, use_cache=False).logits
            loss, _ = combined_loss(student_logits, teacher_logits, batch, alpha_kd)

            prox = student_logits.new_zeros(())
            for module, target in prox_targets:
                prox = prox + 0.5 * admm_lambda * torch.sum((module.weight - target) ** 2)

            (loss / grad_accum + prox / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return opt


@torch.no_grad()
def _compute_hessian(model: nn.Module, module: nn.Linear, calib_ids: torch.Tensor, seqlen: int, device, damping: float) -> torch.Tensor:
    hess = LayerHessian(module.weight.shape[1], device)
    handle = module.register_forward_pre_hook(hess.hook)
    model.eval()
    batch_size = 4
    for i in range(0, calib_ids.shape[0], batch_size):
        batch = calib_ids[i : i + batch_size].to(device)
        model(input_ids=batch, use_cache=False)
    handle.remove()
    return hess.damped(damping=damping)


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

    cfg.target_sparsity is a fixed reference point every layer warm-starts
    toward, independent of whatever overall sparsity Phase 2 is eventually
    asked for -- Phase 2's own Z-step re-derives its keep-count against the
    real target every round regardless of Phase 1's exact per-layer
    support, so a lower final target simply lets some of these coordinates
    back in during Phase 2. That's what lets Phase 1 run once per model
    and get reused, warm-started, across every Phase 2 sparsity level.
    """
    _set_trainable([pl.module for pl in layers], False)
    states: dict[str, LayerState] = {}
    total_steps = cfg.rounds * cfg.x_steps

    for li, pl in enumerate(layers):
        module = pl.module
        n_out, n_in = module.weight.shape
        x0 = module.weight.data.clone()
        x0_norm = x0.norm().item()

        H = _compute_hessian(model, module, calib_ids, cfg.seqlen, device, cfg.damping)
        k_keep = _k_keep(n_in, cfg.target_sparsity)

        x = x0.clone()
        z = x0.clone()
        u = torch.zeros_like(x0)

        # One optimizer for this layer's whole run, reused across rounds
        # (see _x_step's docstring). Still rebuilt fresh per layer --
        # layers are genuinely separate sub-problems.
        opt = _build_optimizer([module], cfg.lr)

        for r in range(cfg.rounds):
            admm_lambda = _cosine_lambda_ramp((r + 1) * cfg.x_steps, total_steps, cfg.admm_lambda_max)
            module.weight.data.copy_(x)
            _x_step(
                model, teacher, [module], [(module, (z - u))],
                calib_ids, cfg.x_steps, cfg.micro_batch, cfg.grad_accum,
                cfg.lr, admm_lambda, cfg.alpha_kd, device,
                max_grad_norm=cfg.max_grad_norm, opt=opt,
            )
            x = module.weight.data.clone()

            v = x + u
            z = zstep_fn(v, H, k_keep)
            u = u + x - z

        module.weight.data.copy_(z)
        _set_trainable([module], False)
        actual_sparsity = (z == 0).float().mean().item()
        states[pl.name] = LayerState(
            name=pl.name, module=module, x=x, z=z, u=u, H=H,
            n_in=n_in, n_out=n_out, x0_norm=x0_norm,
            final_sparsity=actual_sparsity,
        )
        log_fn(
            f"[phase1] layer {li+1}/{len(layers)} {pl.name}: DONE, "
            f"target_sparsity={cfg.target_sparsity:.2f} final_sparsity={actual_sparsity:.3f}"
        )

    return states


def run_phase2(
    model: nn.Module,
    teacher: nn.Module,
    phase1_states: dict[str, LayerState],
    calib_ids: torch.Tensor,
    cfg: Phase2Config,
    device,
    log_fn=print,
    zstep_fn=global_diagonal_project,
) -> dict[str, LayerState]:
    """Joint global ADMM, one global sparsity budget shared by every layer.

    Computes a single k_keep_total = round((1-s) * N_total) across the
    whole network and lets zstep_fn decide, every round, which weights --
    from any layer -- are globally the most/least salient. cfg.target_sparsity
    is the network's overall fraction pruned; individual layers' achieved
    fractions are free to differ, and in practice do.
    """
    working: dict[str, LayerState] = {}
    modules = []
    for name, s in phase1_states.items():
        x = s.x.clone()
        z = s.z.clone()
        u = s.u.clone()
        s.module.weight.data.copy_(x)
        working[name] = LayerState(
            name=name, module=s.module, x=x, z=z, u=u, H=s.H,
            n_in=s.n_in, n_out=s.n_out, x0_norm=s.x0_norm,
        )
        modules.append(s.module)

    _set_trainable(modules, True)

    total_params = sum(s.n_in * s.n_out for s in working.values())
    k_keep_total = max(1, round((1.0 - cfg.target_sparsity) * total_params))

    # One optimizer for the whole run (see _x_step's docstring), with a
    # linear LR decay across every step.
    opt = _build_optimizer(modules, cfg.lr)
    total_steps = cfg.rounds * cfg.x_steps
    scheduler = _build_linear_decay_scheduler(opt, total_steps=total_steps)

    for r in range(cfg.rounds):
        if cfg.lambda_schedule == "constant":
            admm_lambda = cfg.admm_lambda_max
        else:
            admm_lambda = _cosine_lambda_ramp((r + 1) * cfg.x_steps, total_steps, cfg.admm_lambda_max)
        prox_targets = [(s.module, (s.z - s.u)) for s in working.values()]
        _x_step(
            model, teacher, modules, prox_targets, calib_ids,
            cfg.x_steps, cfg.micro_batch, cfg.grad_accum, cfg.lr,
            admm_lambda, cfg.alpha_kd, device,
            max_grad_norm=cfg.max_grad_norm, opt=opt, scheduler=scheduler,
        )

        for s in working.values():
            s.x = s.module.weight.data.clone()
        V = {name: s.x + s.u for name, s in working.items()}
        H = {name: s.H for name, s in working.items()}
        Z = zstep_fn(V, H, k_keep_total)

        max_ratio = 0.0
        for name, s in working.items():
            s.z = Z[name]
            s.u = s.u + s.x - s.z
            s.module.weight.data.copy_(s.x)  # keep dense x live until the final commit
            max_ratio = max(max_ratio, s.u.norm().item() / max(s.x0_norm, 1e-12))
        log_fn(
            f"[phase2] round {r+1}/{cfg.rounds} global_target_sparsity={cfg.target_sparsity:.2f} "
            f"k_keep_total={k_keep_total}/{total_params} max ||u||/||x0||={max_ratio:.4f}"
        )

    for name, s in working.items():
        s.module.weight.data.copy_(s.z)
        s.final_sparsity = (s.z == 0).float().mean().item()
    _set_trainable(modules, False)
    return working

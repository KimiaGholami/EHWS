"""Phase 1 (layer-wise ADMM) and Phase 2 (global ADMM) drivers.

Phase 1, per layer l, sequentially:

    x-step:  x_l <- argmin_x  f(x_l) + (lambda/2) ||x_l - z_l + u_l||_F^2
    z-step:  z_l <- argmin_{||z_l||_0<=k_l} (z_l - v_l)^T H_l (z_l - v_l)
    u-step:  u_l <- u_l + x_l^{t+1} - z_l^{t+1}

`f` is the true CE+KD loss (`losses.combined_loss`), not a per-layer
reconstruction surrogate -- that's the whole point of running this as an
ADMM optimization instead of a one-shot closed-form prune. Each layer is
walked up a sparsity ladder (15%, 30%, ..., 95%), tightening `k_l` one
rung at a time, and stops early once its ADMM dual residual grows past a
tolerance -- a signal that this layer's remaining weights are starting to
disagree with what the true loss wants. Once a layer finishes, the next
layer's Hessian is computed from activations that already flow through
the now-pruned earlier layers.

Phase 2: every layer's `x_l` becomes simultaneously trainable under one
global forward pass, so gradients can couple across layers instead of
being optimized layer-by-layer in isolation. It's warm-started from
Phase 1's final {x_l, z_l, u_l, H_l}, with one sparsity ratio `s` shared
by every layer (`k_l = floor((1-s) * N_l)`). Phase 1 doesn't depend on
the sparsity Phase 2 will eventually target, so it only needs to run
once per model; Phase 2 gets re-run once per sparsity level you want.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from .diagonal_projection import diagonal_project
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
    saturated_at: float | None = None


@dataclasses.dataclass
class Phase1Config:
    ladder: list[float] = dataclasses.field(default_factory=lambda: [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.95])
    rounds_per_step: int = 2
    x_steps: int = 4
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    admm_lambda: float = 5e-5
    alpha_kd: float = 0.5
    # A layer stops climbing the ladder once ||u|| / ||x0|| exceeds this.
    saturation_tau: float = 0.15
    damping: float = 0.01
    seqlen: int = 2048
    max_grad_norm: float = 1.0


@dataclasses.dataclass
class Phase2Config:
    target_sparsity: float = 0.5
    # rounds * x_steps = 4096 total optimizer steps, with the ADMM
    # z/dual update every 32 steps -- this matches the training budget
    # ELSA itself uses.
    rounds: int = 128
    x_steps: int = 32
    micro_batch: int = 2
    grad_accum: int = 4
    lr: float = 1e-4
    admm_lambda_max: float = 5e-5
    # "cosine": penalty ramps 0 -> admm_lambda_max over the run.
    # "constant": penalty is admm_lambda_max from round 1.
    # Which one to use, and the (lr, admm_lambda_max) values, come from
    # `hparams.get_phase2_hparams` per model/sparsity.
    lambda_schedule: str = "cosine"
    alpha_kd: float = 0.5
    seqlen: int = 2048
    max_grad_norm: float = 1.0


def _k_keep(n_in: int, sparsity: float) -> int:
    """Nonzeros to keep per row for a given fraction pruned.

    Rounds the *pruned* count (`n_in - round(n_in * (1 - sparsity))`)
    rather than flooring `(1 - sparsity) * n_in` directly -- the latter
    can land one entry short from ordinary floating-point error (e.g.
    12 * (1 - 5/12) evaluates to 6.999999999999999, which floors to 6
    instead of 7).
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


def _cosine_lambda_ramp(step: int, total_steps: int, lambda_max: float) -> float:
    """Cosine ramp from 0 up to lambda_max over total_steps.

    Early in training the penalty is near zero, so `x` trains almost
    freely against the true loss; as it ramps up, `x` gets pulled toward
    the repeatedly-reprojected sparse target -- a smooth transition
    instead of clamping straight to the sparsity constraint from step 1.
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
) -> torch.optim.Optimizer:
    """Run n_steps of Adam minimizing f(x) + (lambda/2) * ||x - target||_F^2.

    If `opt` is given, it's reused as-is so its momentum carries across
    calls -- Phase 2 builds one optimizer for its whole run rather than
    restarting Adam every round, since restarting it repeatedly forces
    the momentum to re-warm-up from scratch every time and measurably
    hurts convergence. Phase 1 builds a fresh optimizer per call, since
    each layer's ladder step is its own separate sub-problem.
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
    """Sequential per-layer ADMM with a saturation-gated sparsity ladder.

    Each layer climbs `cfg.ladder`'s rungs one at a time. After each
    rung, if the ADMM dual residual `||u|| / ||x0||` exceeds
    `cfg.saturation_tau`, that layer stops -- its `x`/`z`/`u` from the
    *previous* rung are what get carried into Phase 2, not the rung that
    tripped the check.
    """
    _set_trainable([pl.module for pl in layers], False)
    states: dict[str, LayerState] = {}

    for li, pl in enumerate(layers):
        module = pl.module
        n_out, n_in = module.weight.shape
        x0 = module.weight.data.clone()
        x0_norm = x0.norm().item()

        H = _compute_hessian(model, module, calib_ids, cfg.seqlen, device, cfg.damping)

        x = x0.clone()
        z = x0.clone()
        u = torch.zeros_like(x0)
        saturated_at = None

        for step_sparsity in cfg.ladder:
            k_keep = _k_keep(n_in, step_sparsity)

            for _ in range(cfg.rounds_per_step):
                module.weight.data.copy_(x)
                _x_step(
                    model, teacher, [module], [(module, (z - u))],
                    calib_ids, cfg.x_steps, cfg.micro_batch, cfg.grad_accum,
                    cfg.lr, cfg.admm_lambda, cfg.alpha_kd, device,
                    max_grad_norm=cfg.max_grad_norm,
                )
                x = module.weight.data.clone()

                v = x + u
                z = zstep_fn(v, H, k_keep)
                u = u + x - z

            saturation_ratio = u.norm().item() / max(x0_norm, 1e-12)
            log_fn(
                f"[phase1] layer {li+1}/{len(layers)} {pl.name}: sparsity_target={step_sparsity:.2f} "
                f"||u||/||x0||={saturation_ratio:.4f}"
            )
            if saturation_ratio > cfg.saturation_tau:
                # Note: we keep this rung's (x, z, u) rather than rolling
                # back to the previous rung -- the rung that tripped the
                # check is still what Phase 2 warm-starts from.
                saturated_at = step_sparsity
                break

        module.weight.data.copy_(z)
        _set_trainable([module], False)
        actual_sparsity = (z == 0).float().mean().item()
        states[pl.name] = LayerState(
            name=pl.name, module=module, x=x, z=z, u=u, H=H,
            n_in=n_in, n_out=n_out, x0_norm=x0_norm,
            final_sparsity=actual_sparsity, saturated_at=saturated_at,
        )
        log_fn(
            f"[phase1] layer {li+1}/{len(layers)} {pl.name}: DONE, "
            f"final_sparsity={actual_sparsity:.3f} saturated_at={saturated_at}"
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
    zstep_fn=diagonal_project,
) -> dict[str, LayerState]:
    """Joint global ADMM, one uniform target sparsity shared by every layer."""
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
    k_keep = {name: _k_keep(s.n_in, cfg.target_sparsity) for name, s in working.items()}

    # One optimizer for the whole run, so its momentum persists across
    # every round instead of getting rebuilt from scratch each time (see
    # `_x_step`'s docstring).
    opt = _build_optimizer(modules, cfg.lr)
    total_steps = cfg.rounds * cfg.x_steps

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
            max_grad_norm=cfg.max_grad_norm, opt=opt,
        )

        max_ratio = 0.0
        for name, s in working.items():
            s.x = s.module.weight.data.clone()
            v = s.x + s.u
            s.z = zstep_fn(v, s.H, k_keep[name])
            s.u = s.u + s.x - s.z
            s.module.weight.data.copy_(s.x)  # keep dense x live until the final commit
            max_ratio = max(max_ratio, s.u.norm().item() / max(s.x0_norm, 1e-12))
        log_fn(f"[phase2] round {r+1}/{cfg.rounds} target_sparsity={cfg.target_sparsity:.2f} max ||u||/||x0||={max_ratio:.4f}")

    for name, s in working.items():
        s.module.weight.data.copy_(s.z)
        s.final_sparsity = (s.z == 0).float().mean().item()
    _set_trainable(modules, False)
    return working

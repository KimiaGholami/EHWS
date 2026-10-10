"""Phase 2 with over-relaxed ADMM -- a genuine update-rule difference from
real ELSA's (and this repo's own baseline) plain ADMM, not just a
hyperparameter change.

Standard ADMM's z/u update (what `ehws.admm.run_phase2` does, matching
real ELSA) uses the fresh x^{k+1} as-is:

    z^{k+1} = proj(x^{k+1} + u^k)
    u^{k+1} = u^k + x^{k+1} - z^{k+1}

Over-relaxation (Boyd et al. 2011, "Distributed Optimization and
Statistical Learning via ADMM", section 3.4.3) instead relaxes x^{k+1}
toward a blend with the *previous* z before it's used in the z/u update:

    x_hat^{k+1} = alpha * x^{k+1} + (1 - alpha) * z^k
    z^{k+1}     = proj(x_hat^{k+1} + u^k)
    u^{k+1}     = u^k + x_hat^{k+1} - z^{k+1}

alpha=1.0 recovers plain ADMM exactly (x_hat == x^{k+1}) -- Boyd's survey
reports alpha in [1.5, 1.8] typically speeds up convergence (fewer
rounds to reach a given residual) for standard ADMM problems; it is not
guaranteed a priori to help *this* particular non-convex, projected
formulation, which is exactly what this experiment tests. This is a real
algorithmic difference from real ELSA's own `ADMMOptimizer`
(ELSA-official/lib/optimizers.py), which does not relax at all.

Only Phase 2's update rule changes here. Phase 1, the Z-step
(diagonal_project), the X-step's loss (combined_loss), and everything
else are reused unmodified from the parent `ehws` package -- this module
adds only what's different.
"""

from __future__ import annotations

import dataclasses

from ehws.admm import (
    LayerState, Phase2Config, _build_linear_decay_scheduler, _build_optimizer,
    _cosine_lambda_ramp, _k_keep, _set_trainable, _x_step,
)
from ehws.diagonal_projection import diagonal_project


@dataclasses.dataclass
class RelaxedPhase2Config(Phase2Config):
    relax_alpha: float = 1.0  # 1.0 = standard ADMM (identical to ehws.admm.run_phase2); [1.5, 1.8] = literature-typical over-relaxation


def run_phase2_relaxed(
    model,
    teacher,
    phase1_states: dict[str, LayerState],
    calib_ids,
    cfg: RelaxedPhase2Config,
    device,
    log_fn=print,
    zstep_fn=diagonal_project,
) -> dict[str, LayerState]:
    """Identical to `ehws.admm.run_phase2` except the z/u update uses the
    over-relaxed iterate `x_hat` (see module docstring) instead of the raw
    x-step output. At `cfg.relax_alpha == 1.0` this is mathematically
    identical to `run_phase2` -- used as a regression check that this fork
    didn't silently change anything else.
    """
    working: dict[str, LayerState] = {}
    modules = []
    for name, s in phase1_states.items():
        x = s.x.clone()
        z = s.z.clone()
        u = s.u.clone()
        s.module.weight.data.copy_(x)
        working[name] = LayerState(
            name=name, module=s.module, x=x, z=z, u=u, H=s.H.to(device),
            n_in=s.n_in, n_out=s.n_out, x0_norm=s.x0_norm,
        )
        modules.append(s.module)

    _set_trainable(modules, True)
    k_keep = {name: _k_keep(s.n_in, cfg.target_sparsity) for name, s in working.items()}

    opt = _build_optimizer(modules, cfg.lr, betas=(cfg.beta1, cfg.beta2))
    total_steps = cfg.rounds * cfg.x_steps
    scheduler = _build_linear_decay_scheduler(opt, total_steps=total_steps)

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
        )

        max_ratio = 0.0
        for name, s in working.items():
            x_new = s.module.weight.data.clone()
            # Over-relaxation: blend the fresh x-step output with the
            # *previous* round's committed z before it feeds the z/u
            # update. alpha=1.0 makes x_hat == x_new exactly (standard
            # ADMM, see module docstring).
            x_hat = cfg.relax_alpha * x_new + (1.0 - cfg.relax_alpha) * s.z
            s.x = x_new
            v = x_hat + s.u
            s.z = zstep_fn(v, s.H, k_keep[name])
            s.u = s.u + x_hat - s.z
            s.module.weight.data.copy_(s.x)  # keep dense x live until the final commit
            max_ratio = max(max_ratio, s.u.norm().item() / max(s.x0_norm, 1e-12))
        log_fn(
            f"[phase2-relaxed alpha={cfg.relax_alpha:.2f}] round {r+1}/{cfg.rounds} "
            f"target_sparsity={cfg.target_sparsity:.2f} max ||u||/||x0||={max_ratio:.4f}"
        )

    for name, s in working.items():
        s.module.weight.data.copy_(s.z)
        s.final_sparsity = (s.z == 0).float().mean().item()
    _set_trainable(modules, False)
    return working

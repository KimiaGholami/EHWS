"""The true-loss objective used by every X-step (slide 40, Eq. 5-6).

    f(x) = (1 - alpha) * L_CE(x) + alpha * L_KD(x)

``L_CE`` is standard next-token cross-entropy on calibration text (no
human labels needed -- self-supervised, exactly like pretraining).
``L_KD`` is the KL divergence between the current (partially pruned)
"student" model's output distribution and the original dense model's
("teacher") output distribution.

Eq. 5 (per-layer, Phase 1) and Eq. 6 (global, Phase 2) share this exact
functional form -- slide 40 is explicit that Eq. 6 is "same form as Eq. 5,
but x = {x_1,...,x_L} now spans every layer". The only thing that differs
between Phase 1 and Phase 2 is *which parameters are trainable* when the
forward pass is run (one layer's weight vs. every prunable layer's weight
simultaneously), not the loss formula itself -- so a single function
covers both.

Slide 40 also notes: "Teacher logits are pre-cached on calibration data
before pruning begins." We deliberately deviate from literal pre-caching:
caching dense logits for 128 sequences x 2048 tokens x a ~32k-50k
vocabulary would need ~25-50 GB even in fp16, which does not fit
comfortably alongside two model copies + activations on a single 40GB
GPU. Instead we keep one frozen dense teacher copy resident and recompute
its logits on the fly per mini-batch (mathematically identical, since the
teacher is deterministic and has no dropout at eval time -- only the
memory/compute trade-off differs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Next-token CE + KD, as in slide 40's Eq. 5/6.

    Args:
        student_logits: (B, T, V) logits from the model currently being
            trained (Phase 1: one layer trainable; Phase 2: all layers).
        teacher_logits: (B, T, V) logits from the frozen dense teacher,
            same calibration batch.
        input_ids: (B, T) token ids the logits were produced from (used
            to build the next-token CE targets).
        alpha: mixing weight in [0, 1] (slide 40).
        temperature: softens both distributions before the KD term
            (Hinton et al. 2015 convention: divide logits by T before
            softmax/log_softmax, then scale the KL by T^2 so gradient
            magnitudes stay comparable across different T -- d(KL)/d(logit)
            scales as 1/T, so the loss needs T^2 to compensate for both the
            student and teacher softening). T=1 (default) reduces exactly
            to the original untempered KD. Not part of real ELSA at all --
            ELSA has no KD term to temper -- so this is purely an EHWS
            addition, only ever applied to this method's own distillation
            signal.
    """
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    # .float(): F.cross_entropy's internal log_softmax needs full-precision
    # headroom the same way the KD path below does (log_softmax/softmax
    # already .float()'d) -- matters once --dtype bf16 is in play.
    ce = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    # kl_div's "batchmean" divides by input.size(0) only. Left as 3D
    # (B, T, V), that divides by B alone -- T (~1000-2000) too few --
    # inflating KD (and its gradient) by ~T x. Flatten to (B*T, V) first
    # so "batchmean" actually averages per token, matching CE's own
    # per-token averaging via F.cross_entropy's default reduction.
    vocab = student_log_probs.size(-1)
    kd = F.kl_div(
        student_log_probs.reshape(-1, vocab),
        teacher_probs.reshape(-1, vocab),
        reduction="batchmean",
    ) * (temperature ** 2)

    loss = (1.0 - alpha) * ce + alpha * kd
    return loss, {"ce": ce.detach(), "kd": kd.detach()}

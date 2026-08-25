"""The training objective used by every X-step.

    f(x) = (1 - alpha) * CE(x) + alpha * KD(x)

CE is standard next-token cross-entropy on calibration text (no human
labels needed -- the label is just the next token, same as pretraining).
KD is the KL divergence between the current, partially pruned model's
output distribution and the original dense model's output distribution,
i.e. distillation from the dense model into its sparse version as it's
being pruned.

We keep one frozen dense copy of the model resident as the teacher and
recompute its logits on the fly for each mini-batch, rather than
pre-caching them. Pre-caching every calibration token's dense logits
would need tens of gigabytes even in fp16, which doesn't comfortably fit
alongside two live model copies on a single 40GB GPU -- and since the
teacher is deterministic (eval mode, no dropout), recomputing its logits
per batch gives the exact same result, just trading some memory for some
extra compute.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, dict]:
    """Next-token CE + KD.

    Args:
        student_logits: (B, T, V) logits from the model currently being
            trained.
        teacher_logits: (B, T, V) logits from the frozen dense teacher,
            same calibration batch.
        input_ids: (B, T) token ids the logits were produced from, used
            to build the next-token CE targets.
        alpha: KD mixing weight in [0, 1].

    Returns:
        (loss, {"ce": ..., "kd": ...}) with the components detached, for
        logging only.
    """
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    # kl_div's "batchmean" reduction divides by dim-0 size only. Left as
    # 3D (B, T, V), that's B alone -- dividing out only the batch and not
    # the sequence length inflates the KD loss (and its gradient) by
    # roughly the sequence length. Flatten to (B*T, V) first so
    # "batchmean" averages per token, matching cross_entropy's own
    # per-token averaging above.
    vocab = student_log_probs.size(-1)
    kd = F.kl_div(
        student_log_probs.reshape(-1, vocab),
        teacher_probs.reshape(-1, vocab),
        reduction="batchmean",
    )

    loss = (1.0 - alpha) * ce + alpha * kd
    return loss, {"ce": ce.detach(), "kd": kd.detach()}

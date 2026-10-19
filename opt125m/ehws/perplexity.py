"""Perplexity evaluation on tokenized (n_chunks, seqlen) tensors.

Standard next-token perplexity: average per-token negative log-likelihood
across every held-out chunk, exponentiated. Matches the WikiText2/C4
protocol described in ELSA's Appendix B.1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_perplexity(model, token_chunks: torch.Tensor, device, batch_size: int = 4) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, token_chunks.shape[0], batch_size):
        batch = token_chunks[i : i + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += shift_labels.numel()
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


@torch.no_grad()
def compute_perplexity_elsa(model, token_chunks: torch.Tensor, device, batch_size: int = 4) -> float:
    """ELSA-official's exact perplexity formula (`lib/eval.py::calculate_ppl`, from Wanda).

    Same per-chunk next-token loss as `compute_perplexity`, but the summed
    NLL is divided by ``n_chunks * seqlen`` rather than by the number of
    predicted tokens ``n_chunks * (seqlen - 1)``, so it reads very slightly
    lower. Use it with `get_c4_eval_elsa` and a float16 model to reproduce
    ELSA's reported numbers.
    """
    model.eval()
    n, seqlen = token_chunks.shape
    total_nll = 0.0
    for i in range(0, n, batch_size):
        batch = token_chunks[i : i + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum",
        ).item()
    return float(torch.exp(torch.tensor(total_nll / (n * seqlen))))

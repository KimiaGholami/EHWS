"""Architecture-agnostic discovery of prunable layers, in execution order.

Phase 1 processes layers one at a time, and each layer's Hessian needs to
be computed from activations that have already passed through every
earlier layer we've pruned. To get the true processing order for any
decoder-only causal LM -- OPT's attention+MLP stack, HGRN's gated
recurrent stack, or anything else built the same way -- without hard
coding per-architecture wiring, we run one real forward pass with a hook
on every candidate nn.Linear that records the order it actually gets
called in. That gives the true topological order regardless of
architecture, including branching (e.g. attention's q/k/v projections all
run before the block combines them into the input for the output
projection).

Only nn.Linear modules that live inside a transformer block are
candidates. That naturally excludes the token embedding and the final
lm_head, which sit outside the block list -- matching standard practice
in this line of work (SparseGPT, Wanda, and friends all leave embeddings
and the output head dense).
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass
class PrunableLayer:
    name: str
    module: nn.Linear
    block_index: int


def get_decoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Locate the list of repeated transformer/recurrent blocks.

    Covers OPT (`model.model.decoder.layers`) and the `model.model.layers`
    convention used by LLaMA and by HGRN (`fla-hub/hgrn-1.3B-100B`).
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported model architecture: could not locate decoder blocks")


def _find_linears(block: nn.Module, prefix: str) -> dict[str, nn.Linear]:
    found = {}
    for name, child in block.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found[full_name] = child
        else:
            found.update(_find_linears(child, full_name))
    return found


@torch.no_grad()
def discover_prunable_layers(model: nn.Module, sample_batch: torch.Tensor) -> list[PrunableLayer]:
    """Return every prunable nn.Linear, in real forward-execution order.

    Args:
        model: the full causal LM, already on the target device.
        sample_batch: a small (batch, seqlen) input_ids tensor used only
            to trace call order -- no gradients, nothing persisted.
    """
    blocks = get_decoder_blocks(model)
    candidates: dict[str, tuple[nn.Linear, int]] = {}
    for b_idx, block in enumerate(blocks):
        for name, lin in _find_linears(block, "").items():
            candidates[f"blocks.{b_idx}.{name}"] = (lin, b_idx)

    call_order: list[str] = []
    handles = []

    def make_hook(qualified_name):
        def hook(module, inputs):
            call_order.append(qualified_name)

        return hook

    for qname, (lin, _) in candidates.items():
        handles.append(lin.register_forward_pre_hook(make_hook(qname)))

    was_training = model.training
    model.eval()
    model(input_ids=sample_batch, use_cache=False)
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    ordered = []
    seen = set()
    for qname in call_order:
        if qname in seen:
            continue
        seen.add(qname)
        lin, b_idx = candidates[qname]
        ordered.append(PrunableLayer(name=qname, module=lin, block_index=b_idx))

    missing = set(candidates) - seen
    if missing:
        raise RuntimeError(f"{len(missing)} candidate Linear layers were never called during the trace: {missing}")
    return ordered

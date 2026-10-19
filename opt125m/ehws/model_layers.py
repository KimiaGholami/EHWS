"""Architecture-agnostic discovery of prunable layers, in execution order.

The Proposed Work's Phase 1 (slide 39) processes "layer l" sequentially,
recomputing each subsequent layer's Hessian only after the current one
finishes (so it sees activations that have already flowed through earlier,
now-pruned, layers). To do this for *any* decoder-only causal LM (OPT's
attention+MLP nn.Linear stack, HGRN's gated-recurrent nn.Linear stack, ...)
without hand-writing per-architecture wiring, we run one real dry-run
forward pass of the whole model with a hook on every candidate
``nn.Linear`` that records the order it was actually *called* in -- that
gives the true topological/sequential order regardless of architecture,
including branching (e.g. attention's q/k/v computed before the block
combines them into the input for o_proj).

Only ``nn.Linear`` modules that live *inside* a transformer block are
candidates -- this naturally excludes the token embedding and the final
``lm_head`` (which sit outside the block list), matching every pruning
paper referenced in the background section (SparseGPT/Wanda/RIA/AWP/CWS
all leave embeddings and the output head dense).
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

    Covers OPT (`model.model.decoder.layers`), and the shared
    `model.model.layers` convention used by LLaMA, and by HGRN/`fla`'s
    HGRNForCausalLM (`model.model.layers`) -- verified directly against
    `fla-hub/hgrn-1.3B-100B`'s module tree.
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


def disable_fused_kernels(model: nn.Module) -> int:
    """Turn off fused-linear kernels that read an nn.Linear's `.weight`
    directly instead of calling its `.forward()` -- e.g. `fla`'s GatedMLP
    (`fuse_swiglu=True` by default), which computes
    `swiglu_linear(gate, y, down_proj.weight, down_proj.bias)` as one
    fused Triton kernel and never invokes `down_proj.__call__`.

    Both `discover_prunable_layers` (below) and `hessian.py`'s Hessian
    accumulation depend on a `forward_pre_hook` firing on every prunable
    layer -- for a layer whose weight is only ever read directly by a
    fused kernel, that hook can never fire, so its true input activations
    are structurally unobservable this way regardless of anything else
    this codebase does. Found on `fla-hub/hgrn-1.3B-100B`: layer discovery
    raised with all 24 `mlp.down_proj` layers "never called during the
    trace". Verified fix is exact, not an approximation: setting
    `fuse_swiglu=False` on every `GatedMLP` makes `forward()` take the
    `self.down_proj(swiglu(gate, y))` branch instead of the fused one
    (same source file, `fla/modules/mlp.py`) -- confirmed bit-identical
    logits (max abs diff 0.0) between the fused and unfused paths on a
    real forward pass through the real model.

    Duck-typed on the `fuse_swiglu` attribute rather than importing `fla`
    specifically, so this keeps working for any architecture using the
    same fused-kernel-with-an-opt-out convention, and does nothing (0
    patched) for architectures that don't have it -- safe to call
    unconditionally on every model this pipeline loads.

    Returns the number of modules patched (for logging).
    """
    n = 0
    for module in model.modules():
        if hasattr(module, "fuse_swiglu"):
            module.fuse_swiglu = False
            n += 1
    return n


@torch.no_grad()
def discover_prunable_layers(model: nn.Module, sample_batch: torch.Tensor) -> list[PrunableLayer]:
    """Return every prunable nn.Linear, in real forward-execution order.

    Args:
        model: the full causal LM (already on the target device). Call
            `disable_fused_kernels(model)` first if the architecture might
            use fused-linear kernels (see that function's docstring) --
            this function has no way to detect a layer whose hook never
            fires versus one that's genuinely unreachable.
        sample_batch: a small ``(batch, seqlen)`` input_ids tensor used
            purely to trace call order; no gradient, no persisted state.
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

"""Evaluate an externally pruned checkpoint (e.g. ELSA-official's saved model) with
this repo's own eval pipeline, so its numbers are directly comparable to
run_pipeline.py's: same WikiText-2/C4 token sets (from --calib-cache), same
perplexity code, same zero-shot harness.

Needed because ELSA's own C4 eval samples a different C4 validation subset
than ours (WikiText-2 is the full test set in both), so its printed C4 PPL is
not comparable to our results.json numbers. Also reports the same model under
ELSA's exact protocol (`*_elsa` keys): ELSA's C4 test set, its perplexity
formula, a slow tokenizer, and the model cast to float16, so numbers are
directly comparable to ELSA's paper and to ELSA-official runs.

    python3 eval_external.py --model facebook/opt-125m \
        --weights /anvil/scratch/.../model.safetensors --out results/<name>
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import fla  # noqa: F401  -- registers HGRN with transformers' Auto* classes
except ImportError:
    pass

from ehws.calibration import cached, get_c4_eval_elsa, get_wikitext2_test_elsa
from ehws.perplexity import compute_perplexity, compute_perplexity_elsa
from ehws.zeroshot import run_zeroshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="base HF model the checkpoint was pruned from")
    p.add_argument("--weights", required=True, help="pruned state dict (.safetensors), or 'dense' for the unpruned base model")
    p.add_argument("--out", required=True)
    p.add_argument("--calib-cache", default="/anvil/scratch/x-kgholami/ehws-calib-cache")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--n-eval-c4", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--skip-zeroshot", action="store_true")
    p.add_argument("--weights-only-elsa", action="store_true",
                   help="skip our-protocol PPL and zero-shot; report only the ELSA-protocol numbers")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    if args.weights != "dense":
        sd = load_file(args.weights)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Tied lm_head is commonly absent from saved state dicts; anything else missing is a real mismatch.
        missing = [k for k in missing if k != "lm_head.weight"]
        if missing or unexpected:
            raise SystemExit(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()

    # Same cache files run_pipeline.py writes/reads (see its _cache_file).
    tag = args.model.replace("/", "_")
    if not args.weights_only_elsa:
        wt2 = torch.load(os.path.join(args.calib_cache, f"{tag}_wt2_seq{args.seqlen}.pt"))
        c4 = torch.load(os.path.join(args.calib_cache, f"{tag}_c4eval_n{args.n_eval_c4}_seq{args.seqlen}_seed{args.seed + 1}.pt"))

    linears = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    zeros = sum((m.weight == 0).sum().item() for m in linears)
    total = sum(m.weight.numel() for m in linears)
    rows_uniform = all(
        (lambda c: c.min() == c.max())((m.weight != 0).sum(1)) for m in linears
    )
    entry = {"weights": args.weights, "achieved_sparsity": zeros / total, "rows_uniform": bool(rows_uniform)}
    if not args.weights_only_elsa:
        entry["wikitext2_ppl"] = compute_perplexity(model, wt2, device)
        entry["c4_ppl"] = compute_perplexity(model, c4, device)
        print(f"achieved_sparsity={entry['achieved_sparsity']:.4f} rows_uniform={rows_uniform} "
              f"wikitext2={entry['wikitext2_ppl']:.3f} c4={entry['c4_ppl']:.3f}", flush=True)

    # ELSA protocol: slow tokenizer, ELSA's C4 set and formula, float16 model.
    slow_tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    wt2_e = cached(os.path.join(args.calib_cache, f"{tag}_wt2_elsa_seq{args.seqlen}.pt"),
                   lambda: get_wikitext2_test_elsa(slow_tok, seqlen=args.seqlen))
    c4_e = cached(os.path.join(args.calib_cache, f"{tag}_c4_elsa_seq{args.seqlen}.pt"),
                  lambda: get_c4_eval_elsa(slow_tok, seqlen=args.seqlen))
    model.half()
    entry["wikitext2_ppl_elsa"] = compute_perplexity_elsa(model, wt2_e, device)
    entry["c4_ppl_elsa"] = compute_perplexity_elsa(model, c4_e, device)
    model.to(getattr(torch, args.dtype))
    print(f"achieved_sparsity={entry['achieved_sparsity']:.4f} rows_uniform={rows_uniform} "
          f"ELSA protocol: wikitext2={entry['wikitext2_ppl_elsa']:.3f} c4={entry['c4_ppl_elsa']:.3f}", flush=True)
    if not args.skip_zeroshot and not args.weights_only_elsa:
        entry["zeroshot"] = run_zeroshot(model, tokenizer, device)
        print(f"zeroshot={entry['zeroshot']}", flush=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(entry, f, indent=2)


if __name__ == "__main__":
    main()

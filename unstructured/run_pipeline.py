#!/usr/bin/env python
"""End-to-end driver: dense baseline -> Phase 1 (once) -> Phase 2 (per target
overall sparsity) -> perplexity + zero-shot evaluation -> JSON results.

`--sparsities` is the network's *overall* target sparsity, not a per-layer
one -- Phase 2 spends one global weight budget across every layer instead
of forcing each layer to that same ratio (see README.md).

Defaults match the settings used to produce this repo's measured result
(OPT-125M, 50% overall sparsity: WikiText2 35.05 / C4 29.80 -- see
README.md).

Example:

    python run_pipeline.py --model facebook/opt-125m --sparsities 0.5 --out results/opt-125m
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import fla  # noqa: F401  -- registers HGRN with transformers' Auto* classes
except ImportError:
    pass

from elgpu.admm import Phase1Config, Phase2Config, run_phase1, run_phase2
from elgpu.calibration import get_c4_calibration, get_c4_eval, get_wikitext2_test
from elgpu.hparams import get_phase2_hparams
from elgpu.model_layers import discover_prunable_layers
from elgpu.perplexity import compute_perplexity
from elgpu.zeroshot import run_zeroshot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
                    help="overall (network-wide) target sparsity -- not applied uniformly per layer")
    p.add_argument("--zeroshot-sparsities", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--n-calib", type=int, default=1024)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--n-eval-c4", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True, help="output directory for results.json")

    # Phase 1 -- one continuous per-layer ADMM run straight at
    # --p1-target-sparsity, penalty cosine-ramped 0 -> --p1-lambda-max.
    p.add_argument("--p1-target-sparsity", type=float, default=0.7)
    p.add_argument("--p1-rounds", type=int, default=14)
    p.add_argument("--p1-x-steps", type=int, default=4)
    p.add_argument("--p1-lr", type=float, default=2e-4)
    p.add_argument("--p1-lambda-max", type=float, default=5e-5)
    p.add_argument("--p1-max-grad-norm", type=float, default=1.0)

    # Phase 2 -- rounds * x_steps = 4096 total optimizer steps, dual/z
    # update every 32 steps.
    p.add_argument("--p2-rounds", type=int, default=128)
    p.add_argument("--p2-x-steps", type=int, default=32)
    p.add_argument("--p2-lr", type=float, default=1e-4)
    p.add_argument("--p2-lambda-max", type=float, default=5e-5)
    p.add_argument("--p2-max-grad-norm", type=float, default=1.0)

    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--alpha-kd", type=float, default=0.5)
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument(
        "--no-auto-hparams", action="store_true",
        help="disable the per-model/per-sparsity (lr, lambda, schedule) lookup in elgpu/hparams.py "
             "and use --p2-lr/--p2-lambda-max with a cosine schedule for every sparsity level instead",
    )
    p.add_argument("--skip-zeroshot", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="tiny settings, for checking the pipeline runs end to end")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.smoke_test:
        args.n_calib, args.seqlen, args.n_eval_c4 = 8, 256, 8
        args.p1_target_sparsity = 0.5
        args.p1_rounds, args.p1_x_steps = 2, 1
        args.p2_rounds, args.p2_x_steps = 2, 1
        args.micro_batch, args.grad_accum = 1, 1

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Building frozen teacher copy (pre-pruning dense weights)")
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    print(f"Loading calibration data: C4, n={args.n_calib}, seqlen={args.seqlen}")
    calib_ids = get_c4_calibration(tokenizer, n_samples=args.n_calib, seqlen=args.seqlen, seed=args.seed)

    print("Loading perplexity eval sets: WikiText-2 test, C4 validation")
    wt2_ids = get_wikitext2_test(tokenizer, seqlen=args.seqlen)
    c4_eval_ids = get_c4_eval(tokenizer, n_samples=args.n_eval_c4, seqlen=args.seqlen, seed=args.seed + 1)

    results = {"model": args.model, "config": vars(args), "dense": {}, "sparsity": {}}

    print("Evaluating dense baseline perplexity")
    t0 = time.time()
    results["dense"]["wikitext2_ppl"] = compute_perplexity(model, wt2_ids, device)
    results["dense"]["c4_ppl"] = compute_perplexity(model, c4_eval_ids, device)
    print(f"  wikitext2={results['dense']['wikitext2_ppl']:.3f} c4={results['dense']['c4_ppl']:.3f} ({time.time()-t0:.1f}s)")

    if not args.skip_zeroshot:
        print("Evaluating dense baseline zero-shot accuracy")
        results["dense"]["zeroshot"] = run_zeroshot(model, tokenizer, device)
        print(f"  {results['dense']['zeroshot']}")

    print("Discovering prunable layers (execution-order trace)")
    layers = discover_prunable_layers(model, calib_ids[:1].to(device))
    print(f"  found {len(layers)} prunable nn.Linear layers")

    print("=" * 30, "PHASE 1: layer-wise ADMM", "=" * 30)
    p1_cfg = Phase1Config(
        target_sparsity=args.p1_target_sparsity, rounds=args.p1_rounds, x_steps=args.p1_x_steps,
        micro_batch=args.micro_batch, grad_accum=args.grad_accum, lr=args.p1_lr,
        admm_lambda_max=args.p1_lambda_max, alpha_kd=args.alpha_kd,
        damping=args.damping, seqlen=args.seqlen, max_grad_norm=args.p1_max_grad_norm,
    )
    t0 = time.time()
    phase1_states = run_phase1(model, teacher, layers, calib_ids, p1_cfg, device)
    print(f"Phase 1 complete in {time.time()-t0:.1f}s")
    results["phase1_summary"] = {
        name: {"final_sparsity": s.final_sparsity}
        for name, s in phase1_states.items()
    }

    for sparsity in args.sparsities:
        print("=" * 30, f"PHASE 2: global ADMM (unstructured), overall target sparsity={sparsity:.2f}", "=" * 30)
        if args.no_auto_hparams:
            lr, lam, schedule = args.p2_lr, args.p2_lambda_max, "cosine"
        else:
            lr, lam, schedule = get_phase2_hparams(args.model, sparsity)
            print(f"  auto hparams (elgpu/hparams.py): lr={lr} lambda={lam} schedule={schedule}")
        p2_cfg = Phase2Config(
            target_sparsity=sparsity, rounds=args.p2_rounds, x_steps=args.p2_x_steps,
            micro_batch=args.micro_batch, grad_accum=args.grad_accum, lr=lr,
            admm_lambda_max=lam, lambda_schedule=schedule, alpha_kd=args.alpha_kd, seqlen=args.seqlen,
            max_grad_norm=args.p2_max_grad_norm,
        )
        t0 = time.time()
        phase2_states = run_phase2(model, teacher, phase1_states, calib_ids, p2_cfg, device)
        print(f"Phase 2 (overall s={sparsity:.2f}) complete in {time.time()-t0:.1f}s")

        achieved = sum(s.final_sparsity * s.n_in * s.n_out for s in phase2_states.values())
        total = sum(s.n_in * s.n_out for s in phase2_states.values())
        layer_sparsities = {name: s.final_sparsity for name, s in phase2_states.items()}
        entry = {
            "target_sparsity": sparsity,
            "achieved_sparsity": achieved / total,
            "wikitext2_ppl": compute_perplexity(model, wt2_ids, device),
            "c4_ppl": compute_perplexity(model, c4_eval_ids, device),
            # Direct evidence of the point of this variant: per-layer
            # sparsity is data-driven, not forced to match the target.
            "layer_sparsity_min": min(layer_sparsities.values()),
            "layer_sparsity_max": max(layer_sparsities.values()),
            "layer_sparsity_stdev": statistics.pstdev(layer_sparsities.values()),
            "layer_sparsities": layer_sparsities,
        }
        print(
            f"  achieved_sparsity={entry['achieved_sparsity']:.4f} "
            f"per-layer range=[{entry['layer_sparsity_min']:.3f}, {entry['layer_sparsity_max']:.3f}] "
            f"wikitext2={entry['wikitext2_ppl']:.3f} c4={entry['c4_ppl']:.3f}"
        )

        if not args.skip_zeroshot and sparsity in args.zeroshot_sparsities:
            entry["zeroshot"] = run_zeroshot(model, tokenizer, device)
            print(f"  zeroshot={entry['zeroshot']}")

        results["sparsity"][str(sparsity)] = entry

        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"Done. Results written to {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()

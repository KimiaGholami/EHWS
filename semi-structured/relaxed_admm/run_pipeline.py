#!/usr/bin/env python
"""Same end-to-end driver as ../run_pipeline.py, except Phase 2 runs
over-relaxed ADMM (admm_relaxed.py) instead of plain ADMM. Phase 1 is
untouched and imported straight from the parent ehws package -- see
admm_relaxed.py's module docstring for what over-relaxation is and why
it's a genuine mathematical difference from real ELSA, not just a
hyperparameter change.

--relax-alpha 1.0 (default) is mathematically identical to the baseline
../run_pipeline.py -- used as a regression check that this fork didn't
silently change anything else.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

# So `import ehws` resolves to ../ehws regardless of cwd this is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import fla  # noqa: F401  -- registers HGRN/etc. architectures with transformers' Auto* classes
except ImportError:
    pass

from admm_relaxed import RelaxedPhase2Config, run_phase2_relaxed

from ehws.admm import (
    Phase1Config, build_dense_states, load_phase1_checkpoint,
    run_phase1, save_phase1_checkpoint,
)
from ehws.calibration import cached, get_c4_calibration, get_c4_eval, get_wikitext2_test
from ehws.diagonal_projection import diagonal_project, magnitude_project
from ehws.hparams import get_phase2_hparams
from ehws.model_layers import discover_prunable_layers, disable_fused_kernels
from ehws.obs_projection import obs_project, obs_select
from ehws.perplexity import compute_perplexity
from ehws.zeroshot import run_zeroshot

ZSTEP_FNS = {
    "diagonal": diagonal_project,
    "magnitude": magnitude_project,
    "obs_select": obs_select,
    "obs_correct": obs_project,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--zeroshot-sparsities", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--n-calib", type=int, default=1024)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--n-eval-c4", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib-cache", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--out", required=True, help="output directory for results JSON")
    p.add_argument("--checkpoint-dir", default=None)

    p.add_argument("--p1-target-sparsity", type=float, default=0.7)
    p.add_argument("--p1-rounds", type=int, default=14)
    p.add_argument("--p1-x-steps", type=int, default=4)
    p.add_argument("--p1-lr", type=float, default=2e-4)
    p.add_argument("--p1-lambda", type=float, default=5e-5)
    p.add_argument("--p1-max-grad-norm", type=float, default=1.0)

    p.add_argument("--p2-rounds", type=int, default=128)
    p.add_argument("--p2-x-steps", type=int, default=32)
    p.add_argument("--p2-lr", type=float, default=1e-4)
    p.add_argument("--p2-lambda-max", type=float, default=5e-5)
    p.add_argument("--p2-lambda-schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--p2-max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--relax-alpha", type=float, default=1.0,
        help="ADMM over-relaxation factor for Phase 2 (see admm_relaxed.py). 1.0 = standard ADMM "
             "(identical to ../run_pipeline.py). Literature-typical over-relaxation is 1.5-1.8 "
             "(Boyd et al. 2011, section 3.4.3) -- not guaranteed to help this non-convex, "
             "projected formulation, which is what this sweep tests.",
    )

    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--p2-micro-batch", type=int, default=None)
    p.add_argument("--p2-grad-accum", type=int, default=None)
    p.add_argument("--alpha-kd", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--no-prox-after-clip", dest="prox_after_clip", action="store_false")
    p.set_defaults(prox_after_clip=True)
    p.add_argument("--no-auto-hparams", action="store_true")
    p.add_argument("--skip-zeroshot", action="store_true")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--zstep", choices=list(ZSTEP_FNS.keys()), default="diagonal")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.smoke_test:
        args.n_calib, args.seqlen, args.n_eval_c4 = 8, 256, 8
        args.p1_rounds, args.p1_x_steps = 2, 1
        args.p2_rounds, args.p2_x_steps = 2, 1
        args.micro_batch, args.grad_accum = 1, 1

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_defused = disable_fused_kernels(model)
    if n_defused:
        print(f"  disabled {n_defused} fused-kernel module(s)")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  gradient checkpointing enabled on the student model")

    print("Building frozen teacher copy (pre-pruning dense weights)")
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    model_tag = args.model.replace("/", "_")

    def _cache_file(name: str) -> str | None:
        if args.calib_cache is None:
            return None
        return os.path.join(
            args.calib_cache,
            f"{model_tag}_{name}_n{args.n_calib if name == 'calib' else args.n_eval_c4}"
            f"_seq{args.seqlen}_seed{args.seed if name == 'calib' else args.seed + 1}.pt",
        )

    print(f"Loading calibration data: C4, n={args.n_calib}, seqlen={args.seqlen}")
    calib_ids = cached(
        _cache_file("calib"),
        lambda: get_c4_calibration(tokenizer, n_samples=args.n_calib, seqlen=args.seqlen, seed=args.seed),
    )

    print("Loading perplexity eval sets: WikiText-2 test, C4 validation")
    wt2_ids = cached(
        os.path.join(args.calib_cache, f"{model_tag}_wt2_seq{args.seqlen}.pt") if args.calib_cache else None,
        lambda: get_wikitext2_test(tokenizer, seqlen=args.seqlen),
    )
    c4_eval_ids = cached(
        _cache_file("c4eval"),
        lambda: get_c4_eval(tokenizer, n_samples=args.n_eval_c4, seqlen=args.seqlen, seed=args.seed + 1),
    )

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

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        scratch = os.environ.get("SCRATCH")
        checkpoint_dir = os.path.join(scratch, "ehws-checkpoints", os.path.basename(os.path.normpath(args.out))) if scratch else args.out
    os.makedirs(checkpoint_dir, exist_ok=True)
    p1_checkpoint_path = os.path.join(checkpoint_dir, "phase1_checkpoint.pt")
    p1_meta = {
        "model": args.model, "dtype": args.dtype, "seed": args.seed,
        "skip_phase1": args.skip_phase1,
        "p1_target_sparsity": None if args.skip_phase1 else args.p1_target_sparsity,
        "n_calib": args.n_calib, "seqlen": args.seqlen, "damping": args.damping,
    }
    t0 = time.time()
    if os.path.exists(p1_checkpoint_path):
        print(f"Found Phase 1 checkpoint at {p1_checkpoint_path} -- loading instead of recomputing")
        phase1_states = load_phase1_checkpoint(
            p1_checkpoint_path, layers, device, p1_meta,
            model, calib_ids, args.seqlen, args.damping,
        )
        print(f"Checkpoint loaded in {time.time()-t0:.1f}s")
        results["phase1_summary"] = "loaded from checkpoint"
    elif args.skip_phase1:
        print("=" * 30, "PHASE 1: SKIPPED (--skip-phase1 cold-start ablation)", "=" * 30)
        phase1_states = build_dense_states(model, layers, calib_ids, args.seqlen, device, args.damping)
        print(f"Cold-start Hessians computed in {time.time()-t0:.1f}s")
        results["phase1_summary"] = "skipped (--skip-phase1 cold-start ablation)"
        save_phase1_checkpoint(phase1_states, p1_checkpoint_path, p1_meta)
        print(f"Checkpoint saved to {p1_checkpoint_path}")
    else:
        print("=" * 30, "PHASE 1: layer-wise ADMM (unmodified, standard)", "=" * 30)
        p1_cfg = Phase1Config(
            target_sparsity=args.p1_target_sparsity, rounds=args.p1_rounds, x_steps=args.p1_x_steps,
            micro_batch=args.micro_batch, grad_accum=args.grad_accum, lr=args.p1_lr,
            admm_lambda_max=args.p1_lambda, alpha_kd=args.alpha_kd, kd_temperature=args.kd_temperature,
            damping=args.damping, seqlen=args.seqlen, max_grad_norm=args.p1_max_grad_norm,
            beta1=args.adam_beta1, beta2=args.adam_beta2, prox_after_clip=args.prox_after_clip,
        )
        phase1_states = run_phase1(model, teacher, layers, calib_ids, p1_cfg, device, zstep_fn=ZSTEP_FNS[args.zstep])
        print(f"Phase 1 complete in {time.time()-t0:.1f}s")
        results["phase1_summary"] = {
            name: {"final_sparsity": s.final_sparsity} for name, s in phase1_states.items()
        }
        save_phase1_checkpoint(phase1_states, p1_checkpoint_path, p1_meta)
        print(f"Checkpoint saved to {p1_checkpoint_path}")

    for sparsity in args.sparsities:
        print("=" * 30, f"PHASE 2 (OVER-RELAXED, alpha={args.relax_alpha}): global ADMM, target sparsity={sparsity:.2f}", "=" * 30)
        if args.no_auto_hparams:
            lr, lam, schedule = args.p2_lr, args.p2_lambda_max, args.p2_lambda_schedule
        else:
            lr, lam, schedule = get_phase2_hparams(args.model, sparsity)
            print(f"  auto hparams (ehws/hparams.py): lr={lr} lambda={lam} schedule={schedule}")
        p2_cfg = RelaxedPhase2Config(
            target_sparsity=sparsity, rounds=args.p2_rounds, x_steps=args.p2_x_steps,
            micro_batch=args.p2_micro_batch if args.p2_micro_batch is not None else args.micro_batch,
            grad_accum=args.p2_grad_accum if args.p2_grad_accum is not None else args.grad_accum,
            lr=lr,
            admm_lambda_max=lam, lambda_schedule=schedule, alpha_kd=args.alpha_kd,
            kd_temperature=args.kd_temperature, seqlen=args.seqlen,
            max_grad_norm=args.p2_max_grad_norm,
            beta1=args.adam_beta1, beta2=args.adam_beta2, prox_after_clip=args.prox_after_clip,
            relax_alpha=args.relax_alpha,
        )
        t0 = time.time()
        phase2_states = run_phase2_relaxed(model, teacher, phase1_states, calib_ids, p2_cfg, device, zstep_fn=ZSTEP_FNS[args.zstep])
        print(f"Phase 2 (s={sparsity:.2f}) complete in {time.time()-t0:.1f}s")

        achieved = sum(s.final_sparsity * s.n_in * s.n_out for s in phase2_states.values())
        total = sum(s.n_in * s.n_out for s in phase2_states.values())
        entry = {
            "target_sparsity": sparsity,
            "achieved_sparsity": achieved / total,
            "wikitext2_ppl": compute_perplexity(model, wt2_ids, device),
            "c4_ppl": compute_perplexity(model, c4_eval_ids, device),
        }
        print(f"  achieved_sparsity={entry['achieved_sparsity']:.4f} wikitext2={entry['wikitext2_ppl']:.3f} c4={entry['c4_ppl']:.3f}")

        if not args.skip_zeroshot and sparsity in args.zeroshot_sparsities:
            entry["zeroshot"] = run_zeroshot(model, tokenizer, device)
            print(f"  zeroshot={entry['zeroshot']}")

        results["sparsity"][str(sparsity)] = entry

        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        if args.save_model:
            save_dir = os.path.join(args.out, f"model-s{sparsity}")
            os.makedirs(save_dir, exist_ok=True)
            from safetensors.torch import save_file
            print(f"  moving model to CPU before saving...", flush=True)
            model.to("cpu")
            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
            model.config.save_pretrained(save_dir)
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            model.to(device)
            print(f"  model saved to {save_dir}", flush=True)

    if os.path.exists(p1_checkpoint_path):
        os.remove(p1_checkpoint_path)
    print(f"Done. Results written to {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""End-to-end driver: dense baseline -> Phase 1 (once) -> Phase 2 (per target
sparsity) -> perplexity + zero-shot evaluation -> JSON results.

See README.md for the full method description, exact reproduction
commands, and measured results tables.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import fla  # noqa: F401  -- registers HGRN/etc. architectures with transformers' Auto* classes
except ImportError:
    pass

from ehws.admm import (
    Phase1Config, Phase2Config, build_dense_states, load_phase1_checkpoint,
    run_phase1, run_phase2, save_phase1_checkpoint,
)
from ehws.calibration import get_c4_calibration, get_c4_eval, get_wikitext2_test
from ehws.diagonal_projection import diagonal_project, magnitude_project
from ehws.hparams import get_phase2_hparams
from ehws.model_layers import discover_prunable_layers, disable_fused_kernels
from ehws.obs_projection import obs_project, obs_select
from ehws.perplexity import compute_perplexity
from ehws.zeroshot import run_zeroshot

ZSTEP_FNS = {
    "diagonal": diagonal_project,    # ELSA's own Diag(H) score, no correction (production default)
    "magnitude": magnitude_project,  # plain |v_i| score, no Hessian at all (real ELSA's actual default)
    "obs_select": obs_select,        # OBS score (w^2/Hinv_jj, full off-diagonal block H), no correction
    "obs_correct": obs_project,      # OBS score + exact joint OBS correction on survivors
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--zeroshot-sparsities", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--n-calib", type=int, default=1024,
                    help="EHWS-U measured this as the second-largest lever after the LR retune "
                         "(128 -> 1024 calibration sequences) -- see ehws/admm.py's module docstring")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--n-eval-c4", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32",
        help="model/teacher/ADMM-state (x,z,u,H) storage dtype. float32 is what every validated "
             "OPT-125M result in this repo used -- keep it for reproducing those. bfloat16 is "
             "needed at ~1.3B+ scale: model+teacher+x+z+u+H in fp32 doesn't fit in 40GB (measured "
             "OOM on HGRN-1.3B). Z-step score ranking and the CE/KD loss still upcast to float32 "
             "internally regardless (see diagonal_projection.py/losses.py) -- only storage shrinks.",
    )
    p.add_argument("--out", required=True, help="output directory for results JSON")
    p.add_argument(
        "--checkpoint-dir", default=None,
        help="directory for phase1_checkpoint.pt (default: $SCRATCH/ehws-checkpoints if $SCRATCH is set, "
             "else --out). Phase 1's checkpoint can run several GB at 1B+ scale (see ehws/admm.py's "
             "save_phase1_checkpoint docstring for a real incident where writing it to the NFS-mounted "
             "home filesystem got the writing job killed mid-save) -- point this at fast local/scratch "
             "storage instead. Only results.json goes to --out.",
    )

    # Phase 1 hyperparameters -- one continuous run per layer straight at
    # --p1-target-sparsity (no ladder; see ehws/admm.py's module docstring
    # for why the saturation-gated ladder that used to be here was removed).
    p.add_argument("--p1-target-sparsity", type=float, default=0.7,
                    help="carried over from EHWS-U's own validated sweep, not independently re-swept here")
    p.add_argument("--p1-rounds", type=int, default=14)
    p.add_argument("--p1-x-steps", type=int, default=4)
    p.add_argument("--p1-lr", type=float, default=2e-4)
    p.add_argument("--p1-lambda", type=float, default=5e-5)
    p.add_argument("--p1-max-grad-norm", type=float, default=1.0)

    # Phase 2 hyperparameters -- rounds*x_steps=4096 steps, x_steps=32 is
    # the dual/z-update interval, matching real ELSA's Table 4 exactly
    # ("Training steps: 4096", "Interval k: 32")
    p.add_argument("--p2-rounds", type=int, default=128)
    p.add_argument("--p2-x-steps", type=int, default=32)
    p.add_argument("--p2-lr", type=float, default=1e-4)
    p.add_argument("--p2-lambda-max", type=float, default=5e-5)
    p.add_argument(
        "--p2-lambda-schedule", choices=["cosine", "constant"], default="cosine",
        help="only used with --no-auto-hparams (auto-hparams' own per-model/per-sparsity table -- "
             "ehws/hparams.py -- already picks the right schedule per entry). Previously hardcoded "
             "to 'cosine' here regardless of what was passed; real ELSA's actual HGRN-1.3B@80%% run "
             "(ELSA-official/results/hgrn-1.3b-0.8/run.log) used 'constant', which a manual "
             "--no-auto-hparams override couldn't express until this flag was added.",
    )
    p.add_argument("--p2-max-grad-norm", type=float, default=1.0)

    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)  # effective batch = micro_batch * grad_accum
    p.add_argument(
        "--p2-micro-batch", type=int, default=None,
        help="override --micro-batch for Phase 2 only (default: same as --micro-batch). Phase 2's "
             "KD-loss computation (float32 log_softmax/softmax over the full vocab, real seqlen) is "
             "the tightest point in the whole pipeline's memory budget at 1B+ scale -- measured OOM "
             "there on HGRN-1.3B even after every other memory fix. Lowering only this (with "
             "--p2-grad-accum raised to match, preserving the same effective batch) fixes it without "
             "slowing down Phase 1, which doesn't need it.",
    )
    p.add_argument("--p2-grad-accum", type=int, default=None, help="override --grad-accum for Phase 2 only")
    p.add_argument("--alpha-kd", type=float, default=0.5)
    p.add_argument(
        "--kd-temperature", type=float, default=1.0,
        help="softens both student/teacher distributions before the KD term (Hinton et al. convention: "
             "divide logits by T, then scale KD by T^2). 1.0 (default) = untempered, original behavior. "
             "Not something real ELSA has any analogue for -- it has no KD term to temper -- so this is "
             "purely an EHWS addition (ehws/losses.py's combined_loss docstring has the full derivation).",
    )
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Adam beta1 for both Phase 1 and Phase 2's optimizer",
    )
    p.add_argument(
        "--adam-beta2", type=float, default=0.999,
        help="Adam beta2. Default 0.999 is PyTorch's stock default -- NOT what real ELSA actually "
             "configures (0.95, via admm_beta2 in ELSA-official/lib/trainer.py). Pass 0.95 to match "
             "real ELSA; beta2=0.999's ~1000-step averaging window barely adapts within Phase 1's "
             "56-step-per-layer runs, unlike ELSA's own tuning assumption.",
    )
    p.add_argument(
        "--no-prox-after-clip", dest="prox_after_clip", action="store_false",
        help="add the ADMM proximal-term gradient inside the backpropped loss (pre-clip) instead of "
             "after clip_grad_norm_. Default (on, matching real ELSA's ADMMOptimizer._proximal_update "
             "exactly -- ELSA-official/lib/optimizers.py's own comment: 'This ensures proximal is not "
             "clipped') keeps only the task loss clipped and adds the proximal gradient after; passing "
             "this flag reverts to the original EHWS behavior of clipping the combined task+proximal "
             "gradient together, which can attenuate the ADMM constraint pull whenever the combined "
             "norm exceeds --p2-max-grad-norm/--p1-max-grad-norm. Made the default 2026-09-11 after "
             "the opt-125m-fixablation-prox-after-clip ablation beat the prior default on both WT2 "
             "(85.98 vs 90.59) and C4 (52.68 vs 53.86) @80%% sparsity -- see PROGRESS.md.",
    )
    p.set_defaults(prox_after_clip=True)
    p.add_argument(
        "--no-auto-hparams", action="store_true",
        help="disable per-model/per-sparsity (lr, lambda, schedule) lookup (ehws/hparams.py) "
             "and use the fixed --p2-lr/--p2-lambda-max/cosine schedule for every sparsity level instead",
    )
    p.add_argument("--skip-zeroshot", action="store_true")
    p.add_argument(
        "--skip-phase1", action="store_true",
        help="cold-start ablation: skip Phase 1 entirely, run Phase 2 directly from the dense model "
             "(x=z=dense weights, u=0) -- isolates whether Phase 1's warm-start is actually helping",
    )
    p.add_argument(
        "--skip-phase2", action="store_true",
        help="Phase-1-only ablation: run Phase 1 straight at --p1-target-sparsity (overriding its "
             "default 0.7 warm-start value to whatever sparsity you actually want evaluated) and "
             "evaluate directly on its output, without a Phase 2 global fine-tune. --sparsities is "
             "ignored in this mode -- the evaluated sparsity is --p1-target-sparsity.",
    )
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="trade compute for activation memory on the student model's forward/backward pass -- "
             "recomputes activations during backward instead of storing them from the forward pass. "
             "Needed at 1B+ scale: measured OOM inside .backward() on HGRN-1.3B even after every "
             "other memory fix (bf16, Hessian offload, foreach=False Adam, smaller --p2-micro-batch). "
             "No effect on results, only speed (more recompute) vs. memory (less storage).",
    )
    p.add_argument("--smoke-test", action="store_true", help="tiny settings, for pipeline validation only")
    p.add_argument(
        "--save-model", action="store_true",
        help="save the pruned model + tokenizer via save_pretrained() after each requested sparsity's "
             "Phase 2 finishes, to <out>/model-s<sparsity>/ -- needed to later push a checkpoint anywhere "
             "(e.g. HuggingFace Hub); no run so far in this repo has persisted actual pruned weights, "
             "only the results.json metrics.",
    )
    p.add_argument(
        "--zstep", choices=list(ZSTEP_FNS.keys()), default="diagonal",
        help="Z-step selection+correction mechanism, used by both Phase 1 and Phase 2 (same "
             "zstep_fn at both call sites, matching how --sparsities/--p1-target-sparsity already "
             "share one mechanism across phases). 'diagonal' (default): ELSA's own Diag(H) score, "
             "survivors unchanged. 'magnitude': plain |v_i| score, no Hessian involved at all -- "
             "matches real ELSA's actual default projection mode; isolates whether 'diagonal's "
             "Hessian weighting itself (not just its damping) is the problem on ill-conditioned "
             "real-data Hessians. 'obs_select': full off-diagonal-block OBS score (w^2/Hinv_jj), "
             "survivors still left unchanged (no correction) -- isolates selection quality from "
             "correction. 'obs_correct': OBS score + exact joint OBS correction on survivors "
             "(ehws/obs_projection.py's obs_project). See obs_projection.py's module docstring for "
             "the full 2x2 this set of three points into.",
    )
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
        print(f"  disabled {n_defused} fused-kernel module(s) (e.g. fla's GatedMLP.fuse_swiglu) "
              f"so every prunable layer's forward_pre_hook actually fires -- see model_layers.py")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  gradient checkpointing enabled on the student model (trades compute for activation memory; "
              "the teacher never needs backward, so it's left as-is)")

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

    # Checkpoint Phase 1's (or build_dense_states') output -- both can take
    # hours at 1B+ parameter scale, and losing that to an unrelated Phase 2
    # crash (a real incident: HGRN-1.3B's Phase 1 took ~4.5h, then Phase 2
    # OOM'd immediately on entering training) means redoing all of it just
    # to retry Phase 2. `meta` ties the checkpoint to the exact run
    # configuration it's valid for -- see save_phase1_checkpoint/
    # load_phase1_checkpoint's docstrings in ehws/admm.py.
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
        # Not used for anything except the checkpoint-compatibility check below,
        # but the H recompute in load_phase1_checkpoint depends on all three
        # matching exactly (they determine calib_ids and the Hessian damping) --
        # a stale checkpoint made under different values must be rejected, not
        # silently reused with a mismatched Hessian.
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
        print("=" * 30, "PHASE 1: layer-wise ADMM", "=" * 30)
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

    if args.skip_phase2:
        print("=" * 30, f"PHASE 2: SKIPPED -- evaluating Phase 1 output directly at target_sparsity={args.p1_target_sparsity:.2f}", "=" * 30)
        achieved = sum(s.final_sparsity * s.n_in * s.n_out for s in phase1_states.values())
        total = sum(s.n_in * s.n_out for s in phase1_states.values())
        entry = {
            "target_sparsity": args.p1_target_sparsity,
            "achieved_sparsity": achieved / total,
            "wikitext2_ppl": compute_perplexity(model, wt2_ids, device),
            "c4_ppl": compute_perplexity(model, c4_eval_ids, device),
        }
        print(f"  achieved_sparsity={entry['achieved_sparsity']:.4f} wikitext2={entry['wikitext2_ppl']:.3f} c4={entry['c4_ppl']:.3f}")
        if not args.skip_zeroshot:
            entry["zeroshot"] = run_zeroshot(model, tokenizer, device)
            print(f"  zeroshot={entry['zeroshot']}")
        results["sparsity"][str(args.p1_target_sparsity)] = entry
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        if os.path.exists(p1_checkpoint_path):
            os.remove(p1_checkpoint_path)  # run finished successfully -- nothing left to resume
        print(f"Done (Phase-1-only ablation). Results written to {os.path.join(args.out, 'results.json')}")
        return

    for sparsity in args.sparsities:
        print("=" * 30, f"PHASE 2: global ADMM, target sparsity={sparsity:.2f}", "=" * 30)
        if args.no_auto_hparams:
            lr, lam, schedule = args.p2_lr, args.p2_lambda_max, args.p2_lambda_schedule
        else:
            lr, lam, schedule = get_phase2_hparams(args.model, sparsity)
            print(f"  auto hparams (ehws/hparams.py): lr={lr} lambda={lam} schedule={schedule}")
        p2_cfg = Phase2Config(
            target_sparsity=sparsity, rounds=args.p2_rounds, x_steps=args.p2_x_steps,
            micro_batch=args.p2_micro_batch if args.p2_micro_batch is not None else args.micro_batch,
            grad_accum=args.p2_grad_accum if args.p2_grad_accum is not None else args.grad_accum,
            lr=lr,
            admm_lambda_max=lam, lambda_schedule=schedule, alpha_kd=args.alpha_kd,
            kd_temperature=args.kd_temperature, seqlen=args.seqlen,
            max_grad_norm=args.p2_max_grad_norm,
            beta1=args.adam_beta1, beta2=args.adam_beta2, prox_after_clip=args.prox_after_clip,
        )
        t0 = time.time()
        phase2_states = run_phase2(model, teacher, phase1_states, calib_ids, p2_cfg, device, zstep_fn=ZSTEP_FNS[args.zstep])
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
            # model.save_pretrained() crashes for fla-hub/HGRN models: transformers'
            # remove_tied_weights_from_state_dict -> _get_tied_weight_keys calls
            # `._tied_weights_keys.keys()` assuming a dict, but flash-linear-attention's
            # HGRNForCausalLM sets it as a list (older transformers convention) ->
            # AttributeError: 'list' object has no attribute 'keys'. HGRN's own config has
            # tie_word_embeddings=False, so there's no real weight aliasing to worry about --
            # save the raw state dict directly instead of going through save_pretrained().
            from safetensors.torch import save_file
            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
            model.config.save_pretrained(save_dir)
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  model saved to {save_dir}")

    if os.path.exists(p1_checkpoint_path):
        os.remove(p1_checkpoint_path)  # every requested sparsity finished successfully -- nothing left to resume
    print(f"Done. Results written to {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()

# Progress log

**Read this file first in any new session.** It is written to be
self-contained: everything needed to continue this work from a fresh
session should be here.

## What this project is

`EHWS/semi-structured` implements a two-phase ADMM pruning method for
causal LMs (Phase 1: per-layer sequential warm-start to a fixed 70%
target; Phase 2: global joint ADMM at the actual target sparsity),
extending [ELSA](https://arxiv.org/abs/2510.01650) (`ELSA.pdf`, parent
repo root) -- see `README.md` in this folder for the full method
write-up. `EHWS/unstructured` and `Extreme_Layer_Global_Pruning*` are
sibling projects in the same parent repo, similar method, different
sparsity pattern / lineage.

**This session's throughline**: get a real, same-model, same-sparsity
comparison between this method and ELSA's *official* code on
HGRN-1.3B @ 80% sparsity, understand why this method's PPL was ~10x
worse than ELSA's, and start correcting for it. Two real infra fixes
(checkpoint size/location, a gradient-checkpointing crash in ELSA's
official code) and several completed experiment runs came out of this --
all documented below in the order they happened.

## Fix #1: Phase 1 checkpoint was several GB, written to slow NFS home, and could kill the job mid-write

**Symptom** (the reason this investigation started): a prior HGRN-1.3B
`dualphase-0.8` run's job died silently, no traceback. Root cause found
by inspecting `Extreme_Layer_Global_Pruning/results/hgrn-1.3b-dualphase-0.8/`:
a **4.4GB `phase1_checkpoint.pt.tmp`** file, last written ~5h47m after
that run's log had gone silent -- the checkpoint write to the
NFS-mounted home filesystem (`zfs.anvil.rcac.purdue.edu:/home`) stalled
and the job was killed mid-write.

**Fix, applied to both `ehws/admm.py` (this folder) and
`Extreme_Layer_Global_Pruning/elgp/admm.py`** (near-identical sibling
code, same bug):
- `save_phase1_checkpoint` no longer persists `H` (the dense
  `d_in x d_in` per-layer Hessian -- the single biggest tensor, ~11GB
  total across HGRN-1.3B's 168 layers even at fp32). `load_phase1_checkpoint`
  recomputes it instead, by replaying the same layer order and
  measuring each layer's Hessian against its pre-ADMM weight, exactly
  like a fresh run does -- one calibration forward pass per layer, not
  a redo of Phase 1's expensive ADMM optimization.
- `z` is now saved `.to_sparse()` (lossless -- `diagonal_project` always
  produces exact zeros).
- Fixed a latent bug found while touching this code: the old reload set
  the live module weight to `x` instead of `z`, which would've silently
  evaluated unpruned weights for any `--skip-phase2` run resumed from a
  checkpoint.
- `meta` now also tracks `n_calib`/`seqlen`/`damping` (needed for the H
  recompute to be valid) so a mismatched checkpoint is rejected, not
  silently misused.
- Verified correct with a standalone synthetic round-trip test (tiny
  2-layer toy model): recomputed H matched the original bit-for-bit
  (`0.000e+00` diff), sparse `z` round-tripped exactly, mismatched meta
  correctly rejected.
- `run_pipeline.py` (both folders): new `--checkpoint-dir` flag,
  defaults to `$SCRATCH/ehws-checkpoints/<run-name>` (Anvil:
  `/anvil/scratch/x-kgholami`) instead of writing under `--out` on home.
  Only `results.json` still goes to `--out`.
- Deleted the orphaned 4.4GB `.tmp` file.

**Net effect measured on the actual HGRN-1.3B dual-phase run below**:
checkpoint landed at **11.6GB** on scratch (down from an estimated
17-20GB+ it would have been with `H` included), wrote with no `.tmp`
leftover, no stall.

## Result 1: HGRN-1.3B dual-phase @ 80%, after the checkpoint fix

`results/hgrn-1.3b-dualphase-0.8/results.json` -- completed cleanly,
~9h total (Phase 1 ~5.5h, Phase 2 ~2.9h).

| | WikiText2 PPL | C4 PPL | Zero-shot avg (7-task) |
|---|---|---|---|
| Dense | 11.840 | 16.889 | 0.435 |
| **This method, dual-phase @ 80%** | **529.111** | **262.240** | **0.329** |

Achieved sparsity 0.7999. Launch command (memory-safe flags -- see "Memory
settings that work" below):
```
python run_pipeline.py --model fla-hub/hgrn-1.3B-100B \
  --sparsities 0.8 --zeroshot-sparsities 0.8 \
  --n-calib 1024 --seqlen 2048 --n-eval-c4 256 \
  --micro-batch 2 --grad-accum 4 --p2-micro-batch 1 --p2-grad-accum 8 \
  --dtype bfloat16 --gradient-checkpointing \
  --out results/hgrn-1.3b-dualphase-0.8
```
One transient `CUDACachingAllocator` OOM *warning* fired during the
post-training zero-shot eval transition -- PyTorch's allocator recovered
on its own, not a crash. Not seen since; not fully root-caused, treat as
noise unless it recurs and actually kills a run.

## Setting up ELSA's *official* code for a real same-model comparison

No prior ELSA baseline existed for HGRN-1.3B (it isn't one of ELSA's own
published models -- `ehws/hparams.py` already documented this). Cloned
the real thing: `github.com/log-postech/elsa` ->
**`/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/ELSA-official/`**
(separate git repo, own `.venv` -- **do not** try to reuse this
project's venv, ELSA pins `torch==2.7.0`/`transformers==4.45.0`,
meaningfully different from this repo's `torch==2.13.0`/`transformers==5.14.1`).

**One patch was needed and applied**, in `ELSA-official/lib/utils.py`
(top of file): `import fla` before the `AutoModelForCausalLM` import.
`fla-hub/hgrn-1.3B-100B` ships **no bundled remote-code `.py` files** --
`HGRNForCausalLM`/`HGRNConfig` only exist because the separately
pip-installed `flash-linear-attention` package registers them with
transformers' Auto* classes. ELSA's own loader passes
`trust_remote_code=True` but never imports `fla`, so without this patch
it can't find the architecture at all. `pip install flash-linear-attention`
(same version, 0.5.2, as this project's other venvs) into
`ELSA-official/.venv` was also required.

Verified before committing to a long run: `get_llm('fla-hub/hgrn-1.3B-100B', 2048)`
loads as `HGRNForCausalLM`, `get_model_layers` finds all 24 blocks
(matches the `model.model.layers` convention used by LLaMA/Gemma, which
HGRN also follows), `find_layers` finds all 168 `nn.Linear` layers --
matches this project's own layer count exactly.

**A second real bug was hit and fixed on the first launch attempt**:
`--admm_gradient_checkpointing=True` crashed on step 0 with
`RuntimeError: Trying to backward through the graph a second time` --
gradient checkpointing's recompute-based backward doesn't compose safely
with HGRN's fused/triton custom-autograd kernels from `flash-linear-attention`.
**Fix: just don't pass `--admm_gradient_checkpointing`** (defaults
False) -- ELSA's own README example runs LLaMA-2-**7B** (5x bigger than
HGRN-1.3B) with no gradient checkpointing at all, so a 1.3B model was
never going to need it. Confirmed: relaunching without it trained clean
through all 4096 steps.

Also added `--admm_save_inputs=True` on relaunch so the ~35min
single-core C4 tokenization pass (`--admm_steps=4096 --admm_batch_size=1
--admm_gradient_accumulation_steps=8` needs 32,768 tokenized sequences
up front) gets cached to `ELSA-official/dataset/` for any future retry.

## Result 2: ELSA official code, HGRN-1.3B @ 80% sparsity

`ELSA-official/results/hgrn-1.3b-0.8/run.log` -- training + PPL eval
completed cleanly in ~5h15m.

| | WikiText2 PPL | C4 PPL |
|---|---|---|
| **ELSA official, 80% sparsity** | **54.083** | **36.918** |

Achieved sparsity confirmed exactly 0.8000. **Zero-shot eval crashed**
(`ValueError: Feature type 'List' not found`) -- a `datasets==3.6.0` /
`lm_eval==0.4.7` version-incompatibility bug in ELSA's own
`requirements.txt` pins, unrelated to GPU/memory/our patches. Not yet
fixed. Since `--save_model=False` (default -- deliberately skipped to
avoid an extra multi-GB write we didn't need), **getting zero-shot
numbers would require a full retrain**, not just a fixed eval rerun --
the pruned weights aren't kept anywhere. If zero-shot numbers become
needed, either fix the `datasets` pin (likely `pip install "datasets<3"`
or upgrade `lm_eval`) *before* the next launch, or pass
`--save_model=True --admm_save_path=...` to be able to retry eval alone
next time.

Launch command:
```
cd ELSA-official && export CUDA_VISIBLE_DEVICES=0
.venv/bin/python3 main.py --model="fla-hub/hgrn-1.3B-100B" --seqlen=2048 \
  --sparsity_ratio=0.8 --sparsity_type=unstructured --dataset=c4 \
  --admm_steps=4096 --admm_interval=32 \
  --admm_batch_size=1 --admm_gradient_accumulation_steps=8 \
  --admm_lr=2e-4 --admm_lmda=0.01 --admm_precision=bf16 \
  --admm_gradient_checkpointing=False \
  --admm_dual_dtype=bf16 --admm_split_dtype=bf16 --admm_save_inputs=True \
  --admm_save_path=results/hgrn-1.3b-0.8 --save_model=False --eval_zero_shot=True --seed=0
```

## Why is this method ~10x worse than ELSA at the same model/sparsity? Investigation findings

Read ELSA's actual official code (not just the paper) to find real,
verifiable differences -- not just "80% is harder than 50%" (this
method's own OPT-125M/50% result, 37.37 vs ELSA's 34.14, is only ~9%
off; HGRN-1.3B/80% is ~10x off, an asymmetry that needs its own
explanation, not just "extreme sparsity is hard").

**Three confirmed, code-level differences:**

1. **The Z-step criterion isn't what `ehws/diagonal_projection.py`'s
   docstring claims.** That docstring says it "matches real ELSA's own
   Diag(H) simplification... verified against the official ELSA repo."
   Checked `ELSA-official/lib/optimizers.py` directly: with
   `admm_projection_mode="identity"` (**the default**, and what my
   ELSA-official comparison run used), `importance_i = st.get("importance", None)`
   -- and `"importance"` is **never populated anywhere in the
   codebase**. Falls through to plain `|weight|` magnitude thresholding
   in `_proj_impl_dense`. No calibration Hessian is involved at all in
   the default path. The Diag(H)-weighted score only has an analogue
   under `momentum` mode, and even that uses Adam's own `exp_avg_sq`
   (a training-dynamics statistic), not a SparseGPT-style calibration
   Hessian. **This project's own Z-step (`H_ii * v_i^2`) is Hessian-weighted;
   real ELSA's default is not.** This looks like a real
   documentation/understanding bug in this project, not just a
   hyperparameter choice -- worth fixing the docstring's claim
   regardless of what else gets changed.
2. **The training objective is different.** `ELSA-official/lib/trainer.py`'s
   `compute_loss` is plain next-token cross-entropy (`super().compute_loss(...)`,
   the vanilla HF Trainer default) -- no `teacher`/`kd`/`distill`
   anywhere in that file. This project's `f = 0.5*CE + 0.5*KD`
   (`ehws/losses.py`) mixes in a KD term against a frozen dense teacher
   (deep-copied from the model *before* any pruning, teacher logits
   recomputed on the fly per batch rather than pre-cached -- see that
   file's docstring for why). `ehws/hparams.py`'s own docstring already
   flags this exact issue: real ELSA's published lr values were tuned
   for a pure-CE objective, and this method's KD term "dilutes the
   effective gradient magnitude enough that ELSA's own learning rate is
   too conservative."
3. **HGRN-1.3B's hparams are a doubly-unvalidated borrow.**
   `ehws/hparams.py` uses OPT-1.3B's table as the "closest available
   analogue" for HGRN (documented, explicit approximation). The CE+KD
   gradient-dilution fix (**3x lr**) that was actually measured and
   validated -- on OPT-125M/50%, took WikiText2 from 50.99 -> 35.05,
   landing close to ELSA's 34.14 -- was **never applied or re-verified**
   for the OPT-1.3B table (HGRN's stand-in) at any sparsity. The 0.80
   row used as-is: `(lr=1e-3, lambda=5e-5)`, ELSA's raw published value,
   un-retuned.

**Ablation done to test hypothesis "Phase 1's fixed-70%-warm-start,
pushed further to 80%, is the dominant problem" -- ruled out:**

Cold-start (`--skip-phase1`, i.e. Phase 2 only, from the dense model,
no per-layer warm-start) @ 80% sparsity, otherwise identical settings:

| | WikiText2 PPL | C4 PPL | Zero-shot avg |
|---|---|---|---|
| Dual-phase (P1@70% -> P2@80%) | 529.111 | 262.240 | 0.329 |
| **Cold-start (Phase 2 only)** | **594.048** | **286.829** | **0.338** |

Cold-start is *slightly worse*, not better -- consistent with this
project's own OPT-125M/50% ablation table (cold-start 38.92 vs
dual-phase 37.37, same direction). **Phase 1 warm-start is not the
dominant cause of the gap to ELSA.** That leaves differences #2 and #3
above (loss composition + untuned lr) and #1 (Z-step criterion) as the
more likely dominant factors.

Launch command (same as dual-phase above, plus `--skip-phase1`, output
to `results/hgrn-1.3b-phase2coldstart-0.8`).

## Options discussed for the next ablation (user chose: just retune lr/lambda)

Presented four options; explicitly flagged that pure lr retuning might
not be enough on its own, since this method's HGRN run already used
`lr=1e-3` -- *5x higher* than the `lr=2e-4` ELSA's official run used --
and ELSA still won by 10x, arguing against "just needs a better lr"
being the whole story:
1. Match ELSA's recipe exactly in one run (`alpha_kd=0` AND swap Z-step
   to plain magnitude) -- fastest way to test "can we replicate ELSA's
   result at all", can't isolate which factor mattered.
2. Isolate KD only (`alpha_kd=0`, keep Hessian Z-step, keep lr=1e-3).
3. Isolate Z-step only (keep KD, keep lr=1e-3, swap `diagonal_project`'s
   score to plain `|weight|`).
4. **Chosen: just retune lr/lambda.** Applied the one adjustment with
   actual precedent -- the validated 3x lr fix for CE+KD dilution,
   `1e-3 -> 3e-3`, lambda left at `5e-5` (no precedent to justify moving
   it), cosine schedule unchanged (`--no-auto-hparams --p2-lr 3e-3 --p2-lambda-max 5e-5`).

User initially asked for this as a cold-start run (faster), then said
"wait" and "use both phases not just phase 2" -- so the cold-start-lr3x
attempt was killed before it got far and relaunched as dual-phase.

## Resolved: the Sep 1 `hgrn-1.3b-dualphase-0.8-lr3x` run from the previous write-up

That run actually **died silently ~18 minutes in** (right after the dense
zero-shot eval -- the exact point every pre-checkpoint-fix crash also
happened at), sat dead for over a day with no `results.json` and no one
noticing, and was only discovered and relaunched in the next session (see
"Session 2" below). Its checkpoint dir on scratch was confirmed empty
before relaunch, so nothing was lost. Lesson: a `setsid nohup`-detached
run needs an active check-in to catch a silent early death -- `ps -p <pid>`
alone doesn't distinguish "still running" from "never checked since it
died."

## Memory settings that work for HGRN-1.3B on a 40GB A100 (this project's pipeline)

Every successful run above used:
`--dtype bfloat16 --gradient-checkpointing --p2-micro-batch 1 --p2-grad-accum 8`
(keeps effective batch size at 8, same as `--micro-batch 2 --grad-accum 4`'s
default, just lower peak memory). Phase 2 sits around 35-38GB/40GB with
these -- tight but stable across multiple full runs, not climbing over
time (verified by comparing repeated `nvidia-smi` readings mid-run).
**Without** `--gradient-checkpointing`, Phase 2 genuinely OOMs (see the
original `hgrn-1.3b-phase2coldstart-0.7`/`-0.8` crash logs, pre-dating
this session, real `torch.OutOfMemoryError` at ~39.4/39.5GB).

For **ELSA-official** specifically: do **NOT** pass
`--admm_gradient_checkpointing=True` for HGRN (crashes, see above) --
`--admm_batch_size=1 --admm_gradient_accumulation_steps=8` alone was
sufficient, GPU settled around 30-36GB/40GB, no gradient checkpointing
needed at this model scale per their own README precedent.

## Housekeeping notes

- Runs are always launched fully detached (`setsid nohup ... < /dev/null &
  disown`, pinned to one GPU via `CUDA_VISIBLE_DEVICES`) so a session
  disconnect can't kill a multi-hour job. Get the real PID via
  `pgrep -af "<venv path>/bin/python3.*run_pipeline.py"` afterward --
  `$!` right after `setsid nohup ... &` is unreliable (may capture an
  intermediate shell, not the final detached process).
- `run.log` output is heavily buffered when redirected to a file under
  `nohup` -- it can look "stuck" for hours while the process is actively
  training. Cross-check with `ps -p <pid> -o etime` (alive?) and
  `nvidia-smi` (GPU actually active?) rather than trusting log staleness
  alone.
- A few stray `tail -n0 -F <old run>.log` processes from earlier
  monitoring sessions were left running in the background (harmless,
  just idle file watchers) -- not cleaned up, don't be alarmed if seen
  in `ps aux`.

## Session 2 (Sep 2-3): environment fix, Z-step ablation (rejected), the real lr-retune story, KD temperature, `--save-model`

**Environment**: `EHWS/semi-structured` has no `.venv` of its own and
this cluster's `pip` cannot reach a modern PyPI index (tried and failed).
`Extreme_Layer_Global_Pruning_Unstructured/.venv` has the exact right
versions -- its `requirements.txt` is byte-identical to this folder's --
so every command below uses
`Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3` instead of
trying to create a new venv here.

### New: `--zstep {diagonal,obs_select,obs_correct}`, tried, and rejected back to `diagonal`

Added `ehws/obs_projection.py::obs_select` (full off-diagonal-block OBS
selection score, `w^2/Hinv_jj`, but survivors left unchanged -- no
correction, unlike the existing `obs_project`) and wired both into
`run_pipeline.py` as a `--zstep` flag alongside the existing default
`diagonal`. Motivation: differentiate the Z-step's *selection* criterion
from ELSA's own Diag(H) choice while keeping ADMM's iterative X-step (not
a one-shot Hessian correction) as the actual "correction" mechanism --
the user's explicit design preference.

**Measured on `facebook/opt-125m` @ 80%/90% sparsity (dense-eval-matched,
default lr, before the lr retune below):**

| Z-step | WT2 @80% | C4 @80% | WT2 @90% | C4 @90% |
|---|---|---|---|---|
| `diagonal` (ELSA's own) | 302.2 | 133.3 | 3784.2 | 1437.1 |
| `obs_select` | 439.5 | 186.8 | 5814.5 | 2701.7 |
| `obs_correct` | 3410.9 | 1514.0 | (not run, abandoned) |

Both OBS variants lost to plain `diagonal` at every sparsity tested,
`obs_correct` catastrophically (~11x worse). Root-caused, not just
observed: `diagonal_project`'s `H_ii*v_i^2` score is the *exact* optimal
solution to the no-correction-applied objective (its own docstring
derives this) -- OBS's score is derived assuming a correction step
follows, so pairing it with no correction is a mismatched combination,
not just a weaker heuristic. `obs_correct`'s blowup is most likely the
same real-data Hessian ill-conditioning (~1e4-3e4 condition numbers,
already documented in this file's own header comment) hitting the small
per-block alive-submatrix inversion, whose ridge term (`eps=1e-6`) is
explicitly "for float safety" only, nowhere near enough regularization
-- not independently confirmed with a bumped `eps`, since the user
dropped the whole OBS-correction direction before that check was run
(their own instinct -- ADMM's iterative X-step is a better correction
mechanism than one-shot Hessian redistribution -- is itself a legitimate,
statable design position, and this table is the ablation evidence for
it). **Decision: `diagonal` stays the production default.** The code for
`obs_select`/`obs_correct` is kept (useful ablation-table evidence for
the paper) but not used going forward.

### The real cause of the OPT-125M-vs-ELSA gap: untuned lr at high sparsity, not the Z-step

Checked ELSA's own paper directly (`ELSA.pdf`, Section 5.1 + Table 5):
published C4 PPL for OPT-125M @ 80% is **47.45** (WikiText2 not broken
out at this exact cell, but Fig. 2 shows ELSA nearly flat from 60-90%).
`ehws/hparams.py`'s OPT-125M table transcribes ELSA's Table 5 exactly
(verified cell-by-cell) and the LR-schedule shape (`_build_linear_decay_scheduler`,
matching Table 4's "Linear decay") was already correctly implemented.
So the Z-step and LR-schedule-shape were never the problem. What was:
the validated CE+KD-dilution fix (3x lr, proven at 50% sparsity: 50.99
-> 35.05 WikiText2) had only ever been applied to the hparams table's
0.50 row -- **0.70/0.80/0.90 were still ELSA's raw, untuned values.**
Retuned all three the same way (3x): 0.70 `1e-4->3e-4`, 0.80/0.90
`2e-4->6e-4` (lambda untouched, no precedent to move it).

**Measured on OPT-125M @ 80%/90%, C4 PPL (ELSA target: 47.45):**

| Config | C4 @80% | C4 @90% |
|---|---|---|
| No retune (`diagonal`) | 133.3 | 1437.1 |
| Retuned lr alone (`diagonal-lr3x`) | 65.9 | 133.0 |
| Retuned lr + `n_calib=2048` | **53.9** | **104.8** |
| Retuned lr + `p1-target-sparsity=0.85` | 139.1 (**worse**) | 348.5 (**worse**) |

Two real findings: (1) the lr retune is the dominant lever, closing most
of the gap on its own; (2) doubling calibration (1024->2048) closes more
of it on top, landing within ~13% of ELSA's published number; (3) raising
Phase 1's warm-start target from its untested-at-this-pairing default
(0.7, borrowed from a sibling project's *different* Phase-2 target) to
0.85 made things **worse**, not better -- rejected, keep the 0.7 default.

### HGRN-1.3B: the 3x lr fix does NOT transfer -- it makes HGRN worse

This is the load-bearing surprise of this session. Same single-variable
test (`p2_lr` 1e-3 -> 3e-3, everything else identical to Result 1's
launch command) on HGRN-1.3B @ 80%:

| Config | WT2 @80% | C4 @80% |
|---|---|---|
| Original (`lr=1e-3`, un-retuned, "Result 1" above) | 529.1 | 262.2 |
| **`lr=3e-3` (lr3x) alone** | **1131.6** | **393.7** (worse!) |
| `lr=3e-3` + `n_calib=2048` | 422.1 | 161.9 (better than both above) |
| ELSA official (target) | 54.1 | 36.9 |

So the OPT-125M-validated "3x lr" fix is architecture-specific, not a
universal correction for the CE+KD dilution -- it actively hurts HGRN at
3x, while a larger calibration set (independently) helps HGRN enough to
both offset that damage and beat the original baseline. Two live,
unresolved hypotheses for *why* 3x overshoots on HGRN specifically: (a)
HGRN's gated-linear-recurrent blocks may just be more lr-sensitive than
a transformer's attention/MLP blocks (3x pushes into instability rather
than just faster convergence), or (b) the whole borrowed-OPT-1.3B-table
starting point (`hparams.py`'s documented "closest available analogue"
approximation for HGRN, never itself validated) is off in a way that
scaling in either direction doesn't fix. Not yet distinguished.

**In flight at this write-up** (Sep 3, ~18:00 EDT), launched to
triangulate the real optimum rather than guess a third time:

- `results/hgrn-1.3b-dualphase-0.8-autolr-ncalib2048` (GPU 0): lr
  UNCHANGED (auto-hparams table value, 1e-3) + `n_calib=2048` --
  isolates calibration's effect with the lr variable held at its
  original, known-not-broken value.
- `results/hgrn-1.3b-dualphase-0.8-lr2x` (GPU 1): `lr=2e-3` (2x, not
  3x) + default calibration -- triangulates the lr direction with a
  smaller step between known-good 1x and known-bad 3x.
- `results/opt-125m-extreme-ncalib2048-nolr3x` (GPU 3): OPT-125M,
  `n_calib=2048` alone, no lr retune -- completes the factorization of
  whether the winning 53.9 C4 OPT-125M number is mostly calibration or
  mostly lr (cheap/fast, informative even though OPT-125M isn't the
  primary target).

**On resume, check first** (same pattern as before, three dirs this time):
```
VENV=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv
pgrep -af "$VENV/bin/python3.*run_pipeline.py"
for d in hgrn-1.3b-dualphase-0.8-autolr-ncalib2048 hgrn-1.3b-dualphase-0.8-lr2x opt-125m-extreme-ncalib2048-nolr3x; do
  D=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured/results/$d
  echo "-- $d --"; [ -f "$D/results.json" ] && cat "$D/results.json" || tail -c 2000 "$D/run.log" | tr '\r' '\n' | tail -20
done
```
Once these land: pick whichever HGRN config wins, and if a clear winner
still isn't obvious, the next natural probe is an intermediate lr between
1e-3 and 2e-3 (not yet tried) combined with `n_calib=2048` (now
established as robustly good on both models) -- calibration size looks
like the safer, more transferable lever of the two; lr needs more care
per-architecture.

### New: KD temperature (`--kd-temperature`, default 1.0 = unchanged)

`ehws/losses.py::combined_loss` now takes a `temperature` arg (Hinton et
al. convention: soften both student/teacher logits by `T` before the KD
KL term, scale the KL by `T^2`). Threaded through `Phase1Config`/
`Phase2Config`/`_x_step`/`run_pipeline.py`. Smoke-tested only (T=1.0
reproduces the untempered loss exactly, T=2.0 changes it as expected) --
**not yet run in any real experiment.** Something real ELSA structurally
cannot have (no KD term to temper at all), so it's a clean point of
differentiation regardless of whether it improves PPL -- worth an
OPT-125M sweep (T in {1, 2, 4}) once the HGRN lr/calibration question
above is settled and a GPU is free for a lower-priority experiment.

### New: `--save-model`

`run_pipeline.py` now saves the pruned model + tokenizer via
`save_pretrained()` to `<out>/model-s<sparsity>/` when `--save-model` is
passed. **No run in this repo, ever, has persisted actual pruned
weights before this** -- every result so far is metrics-only
(`results.json`). Needed before anything can be pushed to HuggingFace
Hub or reused downstream. Not yet used in any launched run -- once the
best HGRN config from the in-flight triangulation above is identified,
relaunch it (or a fresh run with the winning config) with `--save-model`
added to actually produce a checkpoint.

### Pending: HuggingFace upload of the best HGRN @ 80% checkpoint

User wants the eventual best semi-structured HGRN-1.3B @ 80% pruned
model pushed to HuggingFace, replacing "previous weights" already there.
**Searched this entire repo (code, READMEs, scripts) for any existing HF
repo_id / push_to_hub / HfApi reference and found none** -- there's no
record here of what "the previous weights on HuggingFace" refers to or
what repo they'd be under.

Listed the account (`ikimyaii`, authenticated via user-supplied token) --
it holds ~120 model repos from a much broader pruning research history
(AWP, RIA, SparseGPT, Wanda, OBS-cancel-block, LoRA-distilled variants,
etc.) that are unrelated to this specific ADMM/EHWS project, so none was
an obvious guess. **User explicitly confirmed the target repo:
`ikimyaii/hgrn-1.3B-nonuniform-80pct`.** Not yet touched -- waiting on
the in-flight HGRN triangulation (see above) to identify the actual best
config before generating a checkpoint (`--save-model`) and uploading.
Note: this repo's current contents haven't been inspected yet (what
"previous weights" are actually in it, what architecture/format) --
worth a quick check before overwriting, not just deleting blind.

### Repo hygiene (unchanged, still true)

`EHWS/`, `ELSA-official/`, and `Extreme_Layer_Global_Pruning/` remain
**untracked** in git as of this session (`git status` in the parent repo
root confirms) -- still no version-control safety net under any of this
session's work, on top of the prior session's same finding.

## Session 3 (Sep 5): the Sep 3 triangulation runs never ran, HF repo cleanup, best-config retrain launched

**Discovered on resume**: the three "in flight" experiments from the end
of Session 2 never actually produced results.
`opt-125m-extreme-ncalib2048-nolr3x` did complete (see below), but both
HGRN runs (`hgrn-1.3b-dualphase-0.8-autolr-ncalib2048`,
`hgrn-1.3b-dualphase-0.8-lr2x`) died within ~20 minutes, right after the
**dense**-model zero-shot eval finished -- before Phase 1 training even
started. No `results.json`, no traceback, process gone, sat dead
2 days unnoticed. Investigated two candidate root causes: (a) every job
in this environment (including this one) is allocated only **1 CPU
core** (`nproc`=1, confirmed via `sacct`/`scontrol`, despite 4 GPUs +
480GB RAM) -- a plausible deadlock/starvation trigger at that pipeline
transition, never previously documented; (b) **user's own explanation,
taken as authoritative**: the OnDemand/VSCode session itself likely died
or was restarted around then, killing all child processes, with SLURM's
own accounting (`sacct` showed job `20277808` TIMEOUT at
2026-09-03T22:01:56, ~48h after its start) only reflecting the *formal*
end, not necessarily the moment the interactive session actually died.
Neither is fully confirmed; **operational takeaway regardless**: this
environment's jobs run under a **2-day (48h) walltime**, so any
multi-hour detached run needs enough runway left in the current session
before launching, and should be checked on rather than assumed durable
across a session boundary.

**`opt-125m-extreme-ncalib2048-nolr3x` result** (the one triangulation
run that did finish, isolating calibration size alone, no lr change):

| Config | C4 @80% |
|---|---|
| No retune | 133.3 |
| n_calib=2048 alone (no lr change) | **56.47** |
| lr3x alone | 65.9 |
| lr3x + n_calib=2048 (best combined) | 53.9 |

Calibration size alone recovers almost all of the combined fix's benefit
on OPT-125M -- makes the untested HGRN analogue (calibration alone, lr
left at the known-safe 1e-3) the single most informative pending
experiment, since 3x lr is known to hurt HGRN specifically.

**Relaunched, all 4 GPUs, this session** (`Extreme_Layer_Global_Pruning_Unstructured/.venv`
python, same as always):
- GPU 0: `hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel` -- re-run of
  the current best HGRN config (lr=3e-3, n_calib=2048; WT2 422.1 / C4
  161.9 previously) with `--save-model` added, since **that original run
  never persisted weights** (no `--save-model` passed at the time,
  Phase 1 scratch checkpoint auto-deleted on success) -- needed to
  actually get a checkpoint onto HuggingFace. Command:
  ```
  CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --model fla-hub/hgrn-1.3B-100B \
    --sparsities 0.8 --zeroshot-sparsities 0.8 --n-calib 2048 --seqlen 2048 --n-eval-c4 256 \
    --micro-batch 2 --grad-accum 4 --p2-micro-batch 1 --p2-grad-accum 8 \
    --dtype bfloat16 --gradient-checkpointing \
    --no-auto-hparams --p2-lr 3e-3 --p2-lambda-max 5e-5 --save-model \
    --out results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel
  ```
- GPU 1: `hgrn-1.3b-dualphase-0.8-autolr-ncalib2048-retry` -- the dead
  Session 2 run, relaunched (auto-hparams lr=1e-3 unchanged, n_calib=2048).
- GPU 2: `hgrn-1.3b-dualphase-0.8-lr2x-retry` -- the other dead Session 2
  run, relaunched (`--no-auto-hparams --p2-lr 2e-3 --p2-lambda-max 5e-5`,
  default n_calib=1024).
- GPU 3: `hgrn-1.3b-dualphase-0.8-alpha-kd0` -- **new**, not run before:
  `--alpha-kd 0.0` (drop the KD term entirely, keep everything else at
  auto-hparams default) to directly isolate whether the CE+KD loss
  composition (ELSA's own objective is pure CE) is a bigger factor than
  any hyperparameter -- this was discussed at the end of Session 2 as
  "Option 2" but the user chose the lr retune instead at the time; worth
  running now given lr/calib tuning alone hasn't closed the gap.

**On resume, check all four**:
```
VENV=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv
pgrep -af "$VENV/bin/python3.*run_pipeline.py"
for d in hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel hgrn-1.3b-dualphase-0.8-autolr-ncalib2048-retry hgrn-1.3b-dualphase-0.8-lr2x-retry hgrn-1.3b-dualphase-0.8-alpha-kd0; do
  D=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured/results/$d
  echo "-- $d --"; [ -f "$D/results.json" ] && cat "$D/results.json" || tail -c 2000 "$D/run.log" | tr '\r' '\n' | tail -20
done
```
Given the Session 2 crash pattern, **do not assume silence means
progress** -- check `ps -p <pid> -o etime` and `nvidia-smi` are both
still showing real activity, not just that `run.log` exists.

### HuggingFace account cleanup and new repo for the EHWS checkpoint

User provided a fresh write-scoped token for `ikimyaii` and directed a
cleanup: **deleted 47 repos** -- all 44 `transformer-1B-*` repos (a much
older, unrelated pruning-method history: awp, ria, sparsegpt, wanda,
obs-cancel-block, etc.) plus all three `hgrn-1.3B-nonuniform-80pct*`
repos (base, `-lora`, `-ft` variants). The base one held a *different*,
non-EHWS nonuniform-boundary-sparsity checkpoint (uploaded 2026-05-11,
WikiText2 PPL 225.4 at ~79.9% sparsity, not a same-method comparison) --
flagged to the user before deletion since it was numerically better than
EHWS's current best, user confirmed deletion anyway.

**New repo created**: `ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct`
(empty, awaiting the GPU-0 retrain above to finish so there's an actual
checkpoint to push -- **upload not yet done**, this is the next action
once `hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel/results.json`
exists and its PPL is sanity-checked against the original 422.1/161.9
before pushing, since it's a fresh training run, not a resume, and could
in principle land on a different result).

Token handling note for future sessions: logged in via
`huggingface_hub.login()` (persists to `~/.cache/huggingface/token`) so
it does not need to be re-supplied or ever passed as a CLI arg (this is
a shared multi-tenant node -- `ps aux` is visible to other users, so
never pass a token as a command-line argument here).

## Session 3 continued: the 4 parallel results, all surprising, plus a real save_pretrained bug

All four Session-3 launches completed (all 4 GPUs used concurrently, no
memory issues -- each independently settled around 12-14GB/40GB, well
under the ~35-38GB ceiling previously measured, likely because
`--save-model`/these particular configs don't all hit Phase 2's absolute
peak simultaneously; not investigated further since nothing was tight).
**Correction (Session 4, 2026-09-09): "~4-4.5h each" above was wrong.**
Measured directly from `hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048/run.log`
(the original winning config's own log, solo/no contention): Phase 1
took 27138.2s (7.54h) and Phase 2 took 14869.2s (4.13h) -- **~11.7h
total** for training alone, before adding calibration-loading and eval
overhead. The "4-4.5h" figure was never sourced from an actual
timestamp and should not be used to estimate completion for this
config; see Session 4 below for how this was caught and what it means
for concurrent-job wall-clock expectations under this node's 1-CPU
constraint.

| Config | WT2 @80% | C4 @80% |
|---|---|---|
| Original best: lr=3e-3 + n_calib=2048 (reproduced) | 422.1 | 161.9 |
| n_calib=2048 alone, lr=1e-3 unchanged (`autolr-ncalib2048-retry`) | **1173.6** | **486.3** |
| lr=2e-3 alone, default calib (`lr2x-retry`) | **2117.2** | **694.8** |
| `alpha-kd=0.0` (drop KD term entirely) | **74511.3** | **29869.6** |

**All three new ablations came out worse than the single-lever baselines
they were meant to improve on -- not the OPT-125M pattern at all.**
Reproduction of the original 422.1/161.9 result was exact (confirms no
seed/nondeterminism issue), so these are real, not noise:

- Calibration size alone (2048, lr untouched) makes HGRN **worse** than
  n_calib=1024 did (529.1 originally) -- the opposite of the OPT-125M
  result where calibration alone did most of the work. **Calibration
  size and lr are not independent, additive levers for HGRN** -- the
  422.1/161.9 win only appears when both move together (3x lr AND 2x
  calib); moving either alone is actively harmful. Not yet understood
  mechanistically.
- lr=2e-3 (2117.2) is *worse* than both lr=1e-3 (529.1) *and* lr=3e-3
  alone (1131.6, Session 2) -- a **non-monotonic** relationship between
  lr and PPL, not a smooth curve to interpolate along. This means the
  lr axis is not safely tunable by simple bisection/triangulation for
  this model -- treat any single untested lr value as unpredictable
  until actually run.
- `alpha_kd=0.0` was **catastrophic** (PPL in the tens of thousands --
  compare dense's 11.8/16.9), not just "worse than with KD." Checked the
  log for a real divergence signature: Phase 1 completed normally (all
  168 layers hit target sparsity exactly), and Phase 2's ADMM residual
  (`max ||u||/||x0||`) ended at 1.52, barely different from the healthy
  run's 1.26 -- **the ADMM optimization itself did not blow up**. The
  likely real explanation: `alpha_kd=0.5`'s hparams (including the 3x lr
  already baked into this launch) were tuned assuming the KD term's
  gradient-diluting effect (`hparams.py`'s own documented reasoning) --
  remove KD and keep the same (already-tripled) lr, and the *undiluted*
  CE gradient is now getting an effectively far-too-large step size,
  plausibly overshooting into a bad basin despite reasonable constraint
  satisfaction. **This is not evidence that KD is structurally required
  to match ELSA** (ELSA's own pure-CE recipe gets WT2 54.1 fine) -- it's
  evidence that dropping KD needs a correspondingly *lower* lr, not the
  same one. The correctly-designed version of this test (still not run):
  `alpha_kd=0.0` paired with ELSA's own actual lr (2e-4, no tripling) or
  at least the untripled auto-hparams default (1e-3), not the
  KD-tuned 3e-3.
- **Net effect on strategy**: the lr/calibration axis has now produced
  5 HGRN data points (1e-3, 2e-3, 3e-3, 3e-3+ncalib2048, ncalib2048-alone)
  with no clean trend -- further single-point guesses on this axis are
  low-value. The KD-ablation result argues for retrying it at an lr
  matched to the no-KD setting rather than abandoning that direction.

### Real bug found and fixed: `--save-model` crashed on every HGRN run

The `hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel` run (meant to
finally produce a checkpoint for the pending HuggingFace upload) hit an
uncaught exception in `model.save_pretrained(save_dir)`:
`AttributeError: 'list' object has no attribute 'keys'`, from inside
transformers' `remove_tied_weights_from_state_dict` ->
`_get_tied_weight_keys`. Root-caused: that function does
`submodule._tied_weights_keys.keys()`, assuming a dict, but
`flash-linear-attention`'s `HGRNForCausalLM` sets `_tied_weights_keys` as
a **list** (the older/standard transformers convention) -- a real
version mismatch between `flash-linear-attention` and this environment's
`transformers` (5.14.1), not a bug in this project's own code. HGRN's own
config has `tie_word_embeddings=False` anyway, so the entire
tied-weights-removal step this crashes inside is unnecessary for this
model. Since `results.json` is written *before* the `--save-model` block
(`run_pipeline.py` line ~336 vs ~339), **the PPL numbers from that run
were still valid and already matched the original 422.1/161.9 result
exactly** -- only the weight-saving step was lost, along with the Phase 1
scratch checkpoint (never got deleted, since the crash happened before
that cleanup line -- harmless leftover, not the full pruned model).

**Fix applied to `run_pipeline.py`'s `--save-model` block**: bypass
`model.save_pretrained()` entirely -- save the raw state dict directly
via `safetensors.torch.save_file`, then `model.config.save_pretrained()`
+ `model.generation_config.save_pretrained()` + `tokenizer.save_pretrained()`
separately. Smoke-tested standalone on the actual dense
`fla-hub/hgrn-1.3B-100B` model (save -> reload from disk -> forward pass
on GPU -> sane loss) before committing to a full ~4h rerun. **This fix is
generic to any fla-hub/HGRN model, not specific to this one run/config**
-- future `--save-model` uses for HGRN should already work.

Relaunched with the fix: `results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel-v2`
(GPU 0, same exact hyperparameters as the original winning config). This
is the one to actually push to HuggingFace
(`ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct`, created empty this
session) once it completes and its PPL is confirmed to match 422.1/161.9
again.

**On resume, check**:
```
VENV=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv
pgrep -af "$VENV/bin/python3.*run_pipeline.py"
D=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured/results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel-v2
[ -f "$D/results.json" ] && cat "$D/results.json" || tail -c 2000 "$D/run.log" | tr '\r' '\n' | tail -20
ls "$D/model-s0.8/" 2>/dev/null  # should show model.safetensors, config.json, tokenizer files -- if missing, the fix didn't take
```

## Current summary (as of this write-up): HGRN-1.3B @ 80% sparsity, EHWS vs ELSA

| | WikiText2 | C4 |
|---|---|---|
| Dense (no pruning) | 11.84 | 16.89 |
| **ELSA official** | **54.08** | **36.92** |
| **EHWS semi-structured, best so far** (lr=3e-3, n_calib=2048) | **422.1** | **161.9** |

Gap: ~7.8x on WikiText2, ~4.4x on C4. `savemodel-v2` (above) is
re-verifying this exact number with `--save-model` now working; not a
new experiment.

**User has explicitly ruled out `alpha_kd=0.0`** (dropping the KD term)
as a direction -- KD stays in the loss. All ideas below keep the
two-phase ADMM + CE+KD method exactly as designed; none of them touch
`alpha_kd` down to zero.

## Next-step menu discussed, not yet started (user has not picked one yet)

Presented five options for closing the gap while keeping the
semi-structured EHWS method and KD in the loss, in priority order:

1. **Non-uniform per-layer sparsity allocation -- highest-ceiling option,
   backed by real (if indirect) evidence.** The now-deleted
   `hgrn-1.3B-nonuniform-80pct` HF checkpoint (a *different*, non-ADMM,
   non-KD method -- boundary sparsities 0.65/0.70/0.75/0.80 tapering to
   0.85 in the middle, `lm_head` at 0.60) got **WikiText2 225.4** at an
   equivalent ~80% overall sparsity -- nearly 2x better than EHWS's
   uniform-80% 422.1. That's a real data point that HGRN specifically
   tolerates non-uniform layer-wise sparsity much better than a flat
   target across every layer. Applying the same *allocation* idea inside
   EHWS's own ADMM+KD machinery (same loss, same two-phase optimization,
   just a per-layer target-sparsity profile instead of one scalar
   `--sparsities` value) has not been tried. **Requires real code
   changes** -- `run_pipeline.py`/`ehws/hparams.py` currently only
   accept a single scalar target sparsity applied uniformly to every
   layer; this would need a per-layer (or per-module-type, e.g. HGRN's
   gating projections vs. attention/MLP projections) sparsity-profile
   mechanism added before it could be tested. Not yet scoped in detail.
2. **Swap the Z-step to plain `|weight|` magnitude, dropping only the
   Hessian weighting** (`ehws/diagonal_projection.py`'s `H_ii * v_i^2`
   score -> plain magnitude). This is the original, still-unresolved
   "difference #1" from the very first gap investigation: ELSA's actual
   default Z-step (`admm_projection_mode="identity"`) has **no Hessian
   involved at all**, while EHWS's does. Doesn't touch KD, doesn't touch
   the ADMM round/x-step structure -- purely changes which weights get
   zeroed each Z-step. Cheapest/fastest option here: needs a new
   `--zstep magnitude` choice added alongside the existing
   `diagonal`/`obs_select`/`obs_correct` options, then one HGRN run.
3. **KD temperature sweep** (`--kd-temperature`, T=2 or T=4,
   `alpha_kd` left at 0.5, untouched). Implemented in `ehws/losses.py`
   since Session 2 but **only smoke-tested, never run in a real
   experiment**. Softens the KD signal without reducing its weight in
   the loss -- a legitimate way to change KD's *behavior* while
   satisfying "keep KD in the loss."
4. **Retune `--damping`** (the Hessian ridge term, currently 0.01,
   documented elsewhere in this codebase as "for float safety only, not
   real regularization"). Given the previously-documented real-data
   Hessian ill-conditioning (~1e4-3e4 condition numbers) and that the
   Z-step's Hessian weighting is exactly the mechanism under suspicion
   in (2), a larger ridge might make that existing Hessian-weighted
   Z-step behave better without abandoning it -- an alternative to (2)
   that keeps the Hessian-weighting design intact.
5. **Intermediate `alpha_kd` values** (0.3, 0.7) instead of only the
   tried 0.5 (and the ruled-out 0.0) -- explores whether a different KD
   weighting (still nonzero) helps, without violating the "keep KD"
   constraint.

**Recommendation given but not acted on**: start with (2) (cheap, fast,
no architecture change, directly tests a specific known hypothesis) in
parallel with properly scoping (1) (highest ceiling, real prior
evidence, but needs real engineering work first). Waiting on user
direction before implementing either.

## Session 4 (2026-09-08/09): savemodel-v2 relaunch, damping sweep, Z-step magnitude ablation implemented, timing correction

### HuggingFace upload retrain (savemodel-v2), relaunched with monitoring

The `savemodel-v2` run from Session 3 (meant to finally produce a
checkpoint for `ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct`) had died
silently on 2026-09-07 (~30min in, no error, no `results.json`) --
another instance of the walltime/silent-death pattern in
`slurm_anvil_environment` memory. Relaunched 2026-09-08 on a fresh SLURM
job (2-day walltime, started 05:35:41) as `results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel-v2`
(PID 1509128, GPU 0), same hyperparameters as the original winning
config (lr=3e-3, n_calib=2048, `--save-model`).

**Monitoring infrastructure added** (`results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel-v2/watchdog/`):
- `watch.sh`: detached (`setsid`+`disown`) background loop, polls every
  5 min for process liveness, GPU 0 memory/utilization, and *new-since-start*
  dmesg Xid/OOM events (explicitly filters out this shared node's
  historical noise from other users' jobs -- a naive `dmesg | grep` will
  otherwise flag old unrelated entries as if they just happened).
- `auto_upload.py`: fires automatically when the process exits and
  `results.json` exists. Refuses to push to HuggingFace unless
  `sparsity.0.8.{wikitext2_ppl,c4_ppl}` are within 15% of the known-good
  422.1/161.9 (this is a fresh training run, not a resume -- could in
  principle land elsewhere) *and* an actual `.safetensors` file exists
  in `model-s0.8/`. Logs to `watchdog/upload.log`.
- HuggingFace auth: logged in via `huggingface_hub.login()` with the
  token passed only as an environment variable to that one Python call,
  never as a CLI argument (this node's `ps aux` is visible to other
  users -- see `slurm_anvil_environment` memory).
- As of this write-up: **still training, nothing uploaded yet**. Check
  `watchdog/{watchdog.log,upload.log}` and
  `HfApi().list_repo_files('ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct')`
  for current state.

### Damping sweep (menu option 4): testing whether a bigger Hessian ridge fixes the Z-step

Launched `--damping` at 0.1 / 1.0 / 10.0 (10x/100x/1000x the default
0.01, documented elsewhere as "for float safety only, not real
regularization") on GPUs 1/2/3, everything else identical to the
current-best config (lr=3e-3, n_calib=2048, KD in the loss at its
default weight). Directories:
`results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-damping{0.1,1.0,10.0}`
(PIDs 1515815/1515818/1515819). A second watchdog
(`results/_watchdog_all/watch_all.sh`) covers these three the same way
(liveness + GPU health + filtered dmesg), no auto-upload (exploratory
only). **As of this write-up: still training, no `results.json` for any
of the three yet.**

### Diagnostic finding: a frozen `run.log` during training is NORMAL, not a hang

All 4 processes above went long stretches (the C4-calibration step: 20min-2h;
Phase 1+2 training: many hours) with **zero new bytes** written to
`run.log`, which repeatedly looked like a stuck/dead process on casual
inspection. Root-caused two distinct (both benign) causes:

1. **`ehws/calibration.py`'s `get_c4_calibration()`** streams C4 documents
   over HTTPS and tokenizes each one on CPU to check length (needs a
   pool of up to `n_samples*20` long-enough docs), with **no progress
   print** at all between "Loading calibration data" and completion.
   Confirmed via `py-spy dump --pid <pid>` showing active work inside
   the tokenizer's `_encode_plus`, and confirmed *actually advancing*
   (not deadlocked) by taking two `py-spy dump --locals` samples ~8s
   apart and checking that the `ids` tensor object address / `text`
   local changed between samples -- a reusable technique for
   distinguishing "genuinely slow" from "actually stuck" when the
   top-level stack frame doesn't visibly change.
2. **`run_pipeline.py`'s Phase 1/Phase 2 print()s are never flushed**
   during training. Every phase/round-transition message uses plain
   `print()`, no `flush=True` anywhere, no `sys.stdout.reconfigure` --
   so on a non-tty (file-redirected, `nohup`/`setsid`) stdout, Python
   fully block-buffers it, and the small volume of these messages (a
   few hundred bytes across all of Phase 1 + all 128 Phase 2 rounds)
   doesn't cross the buffer threshold until much later or process exit.
   **Proof**: in the completed historical run
   (`hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048/run.log`), its mtime is
   bit-for-bit identical to `results.json`'s mtime -- every
   `[phase2] round N/128` line was written in one flush at/near the very
   end, not progressively, despite the file's *content* reading like a
   real-time log.

**The reliable, unbuffered signal to check instead**: the Phase 1
checkpoint file, a genuine filesystem write independent of stdout
buffering -- `$SCRATCH/ehws-checkpoints/<run-name>/phase1_checkpoint.pt`
(`SCRATCH=/anvil/scratch/x-kgholami` on this cluster). Its mtime tells
you exactly when Phase 1 finished (hence when Phase 2 started) even
while `run.log` looks completely frozen. Measured this session: `savemodel-v2`'s
Phase 1 finished at 01:00:29 (started ~13:58 the prior day, so ~11h --
noticeably longer than the historical solo 7.54h, attributable to CPU
contention below); the three damping runs finished Phase 1 between
02:29 and 02:36 (they started ~20min after savemodel-v2, so similar
~12h Phase 1 duration under contention).

**This means the "~4-4.5h each" timing claim earlier in this file (Session
3 section, now corrected in place) was wrong** -- solo Phase 1 alone
took 7.54h historically, not the whole run. Don't use wall-clock elapsed
time or a frozen `run.log` as a failure signal for this pipeline;
use `ps`/`py-spy`/checkpoint-file mtimes/`results.json` existence instead.

### CPU contention: this node's 1-CPU limit makes concurrent-job calibration much slower

Confirmed via `nproc` (=1) and `py-spy`-based CPU-time sampling: running
all 4 jobs concurrently means `get_c4_calibration`'s single-threaded
tokenization loop gets roughly 1/4 of the one available core each,
making that stage take proportionally longer than it would solo. Not a
new problem (Session 3 also ran 4 concurrent jobs successfully), but
worth remembering when estimating how long a concurrent multi-GPU
launch will take versus a solo run -- add real margin, don't assume
linear scaling from a solo timing.

### Implemented: `--zstep magnitude` (menu option 2)

Added `ehws/diagonal_projection.py::magnitude_project()`, matching
`diagonal_project`'s exact call signature but with the `H_ii` Hessian
term dropped entirely (score = plain `v_i^2`, no Hessian anywhere) --
this is real ELSA's actual default projection mode
(`admm_projection_mode="identity"`), unlike EHWS's own `diagonal_project`
which always Hessian-weights. Wired into `run_pipeline.py`'s
`ZSTEP_FNS` registry and `--zstep` CLI choices (now
`diagonal`/`magnitude`/`obs_select`/`obs_correct`). Smoke-tested
standalone (correct shape, correct nonzero-per-row count, argparse
accepts the new choice) but **not yet run as a real HGRN experiment**.

### Next 4 experiments: queued to auto-launch as GPUs free up

Since all 4 GPUs were occupied by the still-running savemodel-v2 +
damping sweep, a queue-launcher was set up instead of waiting idle:
`results/_queue_launcher/launch_next.sh`, detached in the background,
polls every 60s for any of the 4 currently-running PIDs (1509128,
1515815, 1515818, 1515819) exiting. When one does, it checks that GPU's
health (`nvidia-smi` reachable, zero uncorrected ECC errors) before
launching the next queued experiment on it one-for-one; if a GPU fails
the health check, that queue entry is skipped and logged for manual
follow-up rather than launched blind. Queue (all at lr=3e-3,
n_calib=2048, matching the current-best config otherwise, no
`--save-model`):

1. `--zstep magnitude` (menu option 2, now implemented above)
2. `--kd-temperature 2` (menu option 3a)
3. `--kd-temperature 4` (menu option 3b)
4. `--alpha-kd 0.3` (menu option 5a; `0.7` not yet queued -- next
   candidate if this batch doesn't clarify things)

Progress: `results/_queue_launcher/launcher.log`. Each launched run's
own directory follows the naming pattern
`results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-{zstep-magnitude,kd-temp2,kd-temp4,alpha-kd0.3}`.

### Status snapshot at end of session (2026-09-09 ~05:10)

| Run | State |
|---|---|
| savemodel-v2 (GPU 0) | Training (Phase 2), ~15h10m elapsed, no `results.json` yet |
| damping0.1 (GPU 1) | Training (Phase 2), ~14h49m elapsed |
| damping1.0 (GPU 2) | Training (Phase 2), ~14h49m elapsed |
| damping10.0 (GPU 3) | Training (Phase 2), ~14h49m elapsed |
| zstep-magnitude, kd-temp2, kd-temp4, alpha-kd0.3 | Queued, auto-launch on GPU availability |

**Next steps once this batch completes** (not yet decided/started):
- If a damping value or `--zstep magnitude` improves on 422.1/161.9,
  do a finer sweep around the winning value rather than jumping
  elsewhere.
- If none of options 2/3/4/5 move the needle, the highest-remaining-
  ceiling option is still **non-uniform per-layer sparsity allocation**
  (menu option 1) -- real evidence of ~2x improvement from a different,
  non-EHWS method's checkpoint, but needs actual code changes
  (`run_pipeline.py`/`ehws/hparams.py` currently only support one
  scalar sparsity applied uniformly) that have not yet been scoped.
- `alpha_kd=0.7` remains an untested candidate if the `alpha_kd=0.3`
  result is inconclusive.

## Session 4 continued: damping sweep results (all 3 complete) -- inconclusive, gap not closed

| Config | WT2 @80% | C4 @80% |
|---|---|---|
| Baseline: lr=3e-3 + n_calib=2048, damping=0.01 (default) | 422.1 | 161.9 |
| damping=0.1 | **525.8** | **186.5** |
| damping=1.0 | **420.9** | **177.2** |
| damping=10.0 | **468.9** | **171.8** |

**Verdict: damping retuning does not close the gap.** damping=1.0 is a
razor-thin WT2 win (420.9 vs 422.1, ~0.3%, within noise) paired with a
clearly *worse* C4 (177.2 vs 161.9, ~9.5% worse) -- not a real
improvement on both metrics, just a different tradeoff point. damping=0.1
and damping=10.0 are both worse on both metrics than baseline. This adds
a 6th and 7th HGRN data point to the growing pile of non-monotonic
hyperparameter responses on this model (see the lr/n_calib table above) --
**the Hessian ridge term is not the lever that closes this gap.** Option
(4) from the next-step menu is now closed out negatively.

The queue-launcher (`results/_queue_launcher/launch_next.sh`) correctly
auto-launched the next batch as each damping job finished:

| Event time | Freed GPU | Launched | New PID |
|---|---|---|---|
| 08:28:12 (savemodel-v2 done) | 0 | `zstep-magnitude` | 1824865 |
| 09:53:18 (damping1.0 done) | 2 | `kd-temp2` | 1847212 |
| 09:59:20 (damping10.0 done) | 3 | `kd-temp4` | 1849155 |
| 10:05:23 (damping0.1 done) | 1 | `alpha-kd0.3` | 1850612 |

All 4 GPUs now running the queue's remaining batch (option 2's Z-step
ablation, plus options 3/5's KD-temperature and alpha_kd ablations).
Each launch passed its GPU ECC health check first (no failures
encountered). The damping sweep's dedicated watchdog
(`results/_watchdog_all/watch_all.sh`) correctly detected all three
completions and exited cleanly once its watch list was empty -- **no
separate 5-minute-cadence GPU-health watchdog is running for this new
batch of 4** (the queue-launcher only tracks liveness for hand-off
purposes, not full health logging); status checks on these are being
done manually per-session for now. **As of this write-up, all 4 (zstep-magnitude,
kd-temp2, kd-temp4, alpha-kd0.3) still training, no `results.json` for
any yet.**

**Revised next-step priority given the damping result**: since neither
`--damping` retuning (just closed out) nor the still-pending `--zstep
magnitude` result can be assumed to save this gap, **non-uniform
per-layer sparsity allocation (menu option 1) is now the most likely
remaining lever** -- it's the only option with real evidence of a ~2x
improvement (the non-EHWS `hgrn-1.3B-nonuniform-80pct` checkpoint's WT2
225.4). Worth starting to scope its code changes now rather than waiting
for the current batch's KD-temperature/alpha_kd results, since those are
lower-prior-probability wins based on the pattern so far (nothing tried
since the original lr/n_calib tuning has beaten 422.1/161.9 on both
metrics simultaneously).

## Session 4 continued again: zstep-magnitude result -- also negative, option (2) closed out

| Config | WT2 @80% | C4 @80% |
|---|---|---|
| Baseline: `--zstep diagonal` (default, Hessian-weighted) | 422.1 | 161.9 |
| `--zstep magnitude` (plain \|v_i\|, no Hessian -- real ELSA's actual default) | **419.8** | **168.1** |

**Verdict: dropping the Hessian weighting from the Z-step does not close
the gap either.** Same pattern as the damping result: WT2 is a razor-thin,
noise-level improvement (419.8 vs 422.1, ~0.6%) while C4 is clearly worse
(168.1 vs 161.9, ~3.8% worse) -- a different tradeoff point, not a real
win on both metrics. **Option (2) from the next-step menu is now closed
out negatively**, joining option (4) (damping). This is the 8th HGRN data
point showing this pattern: every hyperparameter/mechanism lever tried so
far on the ADMM+KD method itself (lr, n_calib, damping, Z-step Hessian
weighting) either does nothing or trades one metric for the other at the
80.0%-ish achieved-sparsity operating point -- none has beaten 422.1/161.9
on both metrics simultaneously since the original lr=3e-3+n_calib=2048
tuning. This further strengthens the case that **non-uniform per-layer
sparsity allocation (menu option 1) is the one remaining direction with
real prior evidence of a large (~2x) improvement**, since it's a
different kind of lever entirely (changing what gets pruned where, not
how the optimization is tuned) rather than another point on the same
exhausted axis.

The queue-launcher's auto-launched batch is now down to its last 3 items,
still training as of this write-up:

| Job | GPU | Status (as of 2026-09-10 ~05:30) |
|---|---|---|
| `zstep-magnitude` | 0 (now free again) | **Done** -- see table above |
| `kd-temp2` (`--kd-temperature 2`) | 2 | Still training, ~19h39m elapsed |
| `kd-temp4` (`--kd-temperature 4`) | 3 | Still training, ~19h33m elapsed |
| `alpha-kd0.3` (`--alpha-kd 0.3`) | 1 | Still training, ~19h27m elapsed |

No `results.json` yet for the remaining 3. Given the pattern above, these
KD-temperature/alpha_kd ablations (options 3 and 5 on the menu) are
now lower-confidence bets than before the damping/zstep results came in,
but they're already running and cost nothing further to let finish.

**Recommendation reaffirmed**: once the remaining 3 finish, the next real
investment should go into scoping non-uniform per-layer sparsity
allocation (option 1) -- `run_pipeline.py`/`ehws/hparams.py` need a
per-layer (or per-module-type) target-sparsity profile mechanism instead
of the current single scalar `--sparsities` value, which hasn't been
scoped in detail yet. Not started this session.

## Session 5 (2026-09-10/11): kd-temp2/kd-temp4/alpha-kd0.3 relaunch results (options 3/5 closed out), OPT-125M fix-ablation finds a real win, `--prox-after-clip` made the default

**Root cause of the previous session's silent deaths, confirmed**: `kd-temp2`/`kd-temp4`/`alpha-kd0.3` (launched 2026-09-09, see prior session) died from the 48h SLURM walltime expiring mid-Phase-2 on job `20468762` (`TIMEOUT` at 2026-09-10T05:35:45 per `sacct`), never producing a `results.json`. All three had a completed Phase 1 checkpoint on `$SCRATCH`, so a relaunch (`results/_relaunch_kd_alpha_v2/relaunch.sh`, 2026-09-10 18:00) let `run_pipeline.py`'s own checkpoint-resume logic skip Phase 1 entirely and go straight to Phase 2. A `--kd-temperature` wiring bug (unrelated to the timeout) was caught right after the first launch and fixed, and `kd-temp2`/`kd-temp4` were relaunched a second time at 18:14 with the fix in place; `alpha-kd0.3` didn't need the fix (no `--kd-temperature` involved) and kept its 18:00 launch. A watchdog (`results/_relaunch_kd_alpha_v2/watchdog.sh`) polled every 5 min for liveness/GPU/SLURM-time-left; all three finished cleanly overnight, last one at 2026-09-11 07:07.

| Config | WT2 @80% | C4 @80% | vs baseline (422.1/161.9) |
|---|---|---|---|
| `--kd-temperature 2` | 514.2 | 181.2 | worse on both |
| `--kd-temperature 4` | 552.5 | 211.8 | worse on both |
| `--alpha-kd 0.3` | 467.0 | 188.4 | worse on both |

**Verdict: both closed out negatively.** Options (3) and (5) from the next-step menu (KD temperature, alpha_kd) join options (2) and (4) (Z-step, damping) as exhausted levers. Every hyperparameter/mechanism tried directly on the ADMM+KD optimization (lr, n_calib, damping, Z-step Hessian weighting, KD temperature, alpha_kd) has now failed to beat 422.1/161.9 on both metrics simultaneously since the original lr=3e-3+n_calib=2048 tuning — an 11th data point in the same pattern. This leaves non-uniform per-layer sparsity allocation (option 1) as the only remaining item on the original menu with real prior evidence of a large improvement, still not started.

**New thread: auditing EHWS's optimizer mechanics directly against `ELSA-official/`'s actual code, independent of the menu above.** Two divergences found: (a) Adam `beta2` defaults to PyTorch's stock 0.999, not real ELSA's 0.95 (`ELSA-official/lib/trainer.py`'s `admm_beta2`); (b) EHWS adds the ADMM proximal-term gradient into the backpropped loss before `clip_grad_norm_`, so the combined task+proximal gradient gets clipped together, whereas real ELSA's `ADMMOptimizer._proximal_update` (`ELSA-official/lib/optimizers.py`) explicitly adds the proximal gradient *after* clipping ("This ensures proximal is not clipped"). Both were already wired as flags (`--adam-beta2`, `--prox-after-clip`) but never tested. Ablated each in isolation on OPT-125M @80% first (cheap, ~4-7h vs HGRN's ~12-20h) against the existing best OPT-125M baseline (`opt-125m-extreme-lr3x-ncalib2048`: WT2 90.59/C4 53.86), via `results/_opt125m_fix_ablation/launch.sh` (sequential on GPU 3, 2026-09-10 19:04 to 2026-09-11 08:42):

| Config | WT2 @80% | C4 @80% | vs baseline (90.59/53.86) |
|---|---|---|---|
| `--adam-beta2 0.95` | 93.22 | 54.90 | worse on both |
| `--prox-after-clip` | **85.98** | **52.68** | **better on both** (~5.1% WT2, ~2.2% C4) |

**`--prox-after-clip` is the first genuine both-metrics win found in this entire investigation** — everything else (damping, Z-step, KD temperature, alpha_kd, adam-beta2 itself) has been either a wash or a tradeoff. `--adam-beta2 0.95` is closed out negatively.

**Code change: `prox_after_clip` made the default (on) as of this write-up**, since it's a straight improvement with no observed downside and matches real ELSA's actual behavior (not just an ad-hoc knob). Changed in `ehws/admm.py` (`Phase1Config.prox_after_clip`, `Phase2Config.prox_after_clip`, and `_x_step`'s own default, all `False`→`True`) and `run_pipeline.py` (flag flipped from `--prox-after-clip` (`store_true`, default off) to `--no-prox-after-clip` (`store_false`, default on) so the new default can still be reverted for comparison runs). Not yet re-verified against the exact `results.json` numbers above via a rerun — the ablation run itself used the explicit `--prox-after-clip` flag before this default flip, so the recorded 85.98/52.68 numbers stand independent of this code change; the change only affects future runs that don't pass `--no-prox-after-clip`.

**Not yet done**: `--prox-after-clip` has not been tried on HGRN-1.3B yet — only validated on OPT-125M so far. Given it's the first real win found, the natural next run is HGRN-1.3B @80% with the new default (equivalent to the old `--prox-after-clip` flag) against the 422.1/161.9 baseline. Non-uniform per-layer sparsity allocation (option 1) remains unscoped and is still the other live lead.

## Session 5 continued: full EHWS-vs-ELSA-official audit finds the real ADMM lambda has been 200x too small for HGRN this entire investigation, `ehws/hparams.py` fixed

**The single biggest finding of this project to date.** `ELSA-official/results/hgrn-1.3b-0.8/` (a sibling repo on this same machine) contains a complete, un-truncated `run.log` of real ELSA's own code actually being run against `fla-hub/hgrn-1.3B-100B` @80% sparsity — the exact model/sparsity this whole gap-closing effort targets. Its final line: `[('wikitext2', 54.083431243896484), ('c4', 36.917686462402344)]` — the literal target numbers (54.08/36.92) this project has been chasing since the start, produced by a run whose full hyperparameters are logged verbatim at the top of the file. This was not previously read or used by this project; `ehws/hparams.py` instead used OPT-1.3B's *paper-table* row (Table 5, `ELSA.pdf`) as a "closest available analogue" for HGRN, explicitly documented as "an explicit approximation, not a measured value." That approximation was unnecessary — the actual ground truth was sitting one directory tree over the entire time.

**Read against `ehws/admm.py`, `run_pipeline.py`, and the winning HGRN config's own launch command** (`--no-auto-hparams --p2-lr 3e-3 --p2-lambda-max 5e-5`, `--p2-micro-batch 1 --p2-grad-accum 8`, `--dtype bfloat16`), here is every divergence found, and its verdict:

| Parameter | Real ELSA (measured, `run.log`) | EHWS (winning HGRN config) | Verdict |
|---|---|---|---|
| ADMM lambda (`admm_lmda` / `admm_lambda_max`) | **0.01, constant** | **5e-5**, cosine-ramped 0→5e-5 | **200x too small, never isolated as its own ablation — see below** |
| Phase 2 lr | 2e-4 | 3e-3 (15x) | Already known-divergent; tied to the KD-dilution compensation, see below |
| KD term | none at all — plain CE loss, no teacher, no `alpha_kd` | `alpha_kd=0.5` | Already known EHWS-only addition (see Session 3/4); the "proper" ablation (`alpha_kd=0` at real ELSA's *actual* lr) was never run because "real ELSA's actual lr" was itself unknown until now — it's 2e-4, not the previously-assumed 1e-4/1e-3 Phase-2-default guesses |
| Adam beta2 | 0.95 | 0.999 default (0.95 tested in isolation on OPT-125M only, and *with* the still-wrong lambda for that run) | Needs retesting on HGRN with the corrected lambda — the OPT-125M-only negative result doesn't transfer cleanly |
| `admm_interval` / Phase 2 `x_steps` | 32 | 32 | Matches |
| Total ADMM steps | 4096 (`admm_steps`) | 4096 (128 rounds × 32 x_steps) | Matches |
| Phase 2 batch (micro_batch × grad_accum) | 1 × 8 | 1 × 8 (the winning config passed `--p2-micro-batch 1 --p2-grad-accum 8` explicitly) | Matches |
| Training precision | bf16 | bf16 (winning config passed `--dtype bfloat16`) | Matches |
| Projection/Z-step mode | `identity` (plain magnitude, no Hessian weighting) | tested as `--zstep magnitude` (Session 4) — but confounded, see below | Already tested, but at the wrong lambda (see below) |
| Proximal-gradient clip order | after `clip_grad_norm_` | now the default as of this write-up (see above) | **Already fixed this session, independently found and confirmed as a real win** |
| Calibration/training data | 32768 *unique* C4 docs, one epoch, no resampling | `n_calib=2048`, resampled with replacement across 4096×8=32768 draws (~16x reuse per doc) | Real divergence, never isolated as its own ablation |
| Training architecture | one continuous global ADMM run, 0→0.8 sparsity directly | two-phase: per-layer local warm-start to 0.7 (Phase 1, 56 steps/layer) *then* global ADMM to 0.8 (Phase 2) | **Structural, out of scope to change this session** — biggest remaining architectural divergence, flagged for awareness only |

**Why the lambda finding reframes everything from Session 2 onward**: every single HGRN-1.3B data point collected in this project before this session — the original lr/n_calib tuning that found 422.1/161.9, the damping sweep (which retuned a *different* parameter, the Z-step's Hessian-ridge damping, not this ADMM lambda), the Z-step magnitude ablation, KD temperature, alpha_kd — all ran with `admm_lambda_max=5e-5`, cosine-ramped so the *average* pull during training was well under even that. Real ELSA's actual value is a constant 0.01, 200x the EHWS peak. The ADMM lambda controls how hard the proximal term pulls dense weights toward the sparse projection point during training; at 200x too small, that pull is nearly absent for most of training, so the projection at the end effectively works on weights that were never meaningfully constrained toward sparsity-compatibility — a plausible root cause for a good chunk of the whole WT2/C4 gap, and a lever nobody had actually varied in isolation until now (the damping sweep looked similar on the surface but tunes the *Z-step's* Hessian ridge, a different equation entirely — see `ehws/diagonal_projection.py`).

**Code change: `ehws/hparams.py` fixed.** Added a dedicated `_HGRN_1_3B` table (previously HGRN fell through to the `_OPT_1_3B` approximation table via the function's `else` branch). Sparsity=0.80's row now holds the measured ground truth (`lr=2e-4, lambda=0.01, schedule="constant"`) instead of the old approximation (`lr=1e-3, lambda=5e-5, schedule="cosine"`, itself an OPT-1.3B paper-table value, not even the same value the winning `lr3x` config actually used). Every other sparsity (0.3-0.7, 0.9) has no measured ground truth and still falls back to the same OPT-1.3B-approximation values as before — only 0.80 changed. Verified: `py_compile` clean, `get_phase2_hparams('fla-hub/hgrn-1.3B-100B', 0.8)` now returns `(0.0002, 0.01, 'constant')`; other models/sparsities unaffected (spot-checked OPT-125M, OPT-1.3B, an untabulated sparsity, and an unknown model name all still return their previous values).

**Not yet run**: no experiment has used the corrected lambda yet. The natural next experiment is real ELSA's actual recipe run through EHWS's existing Phase 1 + corrected Phase 2 (auto-hparams now supplies it automatically — just drop `--no-auto-hparams --p2-lr --p2-lambda-max` from the launch command): `lr=2e-4, lambda=0.01 constant, alpha_kd=0` (matching real ELSA's no-KD design, now correctly paired with real ELSA's actual lr instead of a KD-diluted-gradient guess), `--adam-beta2 0.95`, `--prox-after-clip` (now default), ideally with `--n-calib` raised well above 2048 to reduce the resampling-reuse divergence. This is effectively "try to reproduce real ELSA's HGRN run through EHWS's own pipeline" rather than another point on the already-exhausted hyperparameter-tweaking axis — a qualitatively different, much better-justified experiment than anything tried in Sessions 2-5 so far. Non-uniform per-layer sparsity allocation (menu option 1) and the Phase1/Phase2 architectural divergence (no analogue in real ELSA at all) remain the other two open structural questions.

## Session 5 continued again: lambda-fix batch results — by far the best HGRN result this project has produced, confirms lambda was the dominant lever

Launched 2026-09-11 ~15:40, all 4 GPUs, per explicit user instruction to **keep KD active** (`alpha_kd=0.5` default, unchanged) in every run — this batch does *not* test real ELSA's no-KD design, only the corrected lambda (`0.01 constant`, see above) combined with varying lr/beta2. All 4 finished cleanly overnight (~10:20-10:26 the next morning, 2026-09-12), no OOM/ECC/crash (watchdog clean throughout — `results/_lambdafix_batch/watchdog.log` only ever logged the expected walltime-countdown warning, nothing else).

| Run | lr | beta2 | WT2 @80% | C4 @80% | vs old best (422.1/161.9) |
|---|---|---|---|---|---|
| `lr2e-4` | 2e-4 (real ELSA's raw lr) | 0.999 default | 207.7 | 125.2 | ~2.0x / ~1.3x better |
| `lr6e-4-kdcomp` | 6e-4 (3x, KD-compensated) | 0.999 default | 131.9 | 71.9 | ~3.2x / ~2.3x better |
| `lr3e-3` | 3e-3 (old best lr, **lambda fix only**) | 0.999 default | 260.2 | 111.2 | ~1.6x / ~1.5x better |
| **`lr6e-4-beta095`** | 6e-4 | **0.95** (real ELSA's) | **85.8** | **57.2** | **~4.9x / ~2.8x better** |

**Every single one of these 4 runs beat the old best (422.1/161.9) on both metrics simultaneously** — the first time that's happened since the original lr=3e-3/n_calib=2048 tuning itself. `lr3e-3` is the cleanest isolation: identical to the old best config in every way except the lambda (5e-5 cosine→0.01 constant), and it alone cuts WT2 by ~40% and C4 by ~31%. This confirms the lambda diagnosis directly rather than just by plausible argument.

**`lr6e-4-beta095` is the new best HGRN-1.3B@80% result by a wide margin — WT2 85.8 / C4 57.2, within ~1.6x/~1.55x of ELSA's own target (54.08/36.92), down from the old best's ~7.8x/~4.4x gap.** This is the corrected lambda + this project's own KD-dilution-compensated lr heuristic (3x real ELSA's 2e-4) + real ELSA's actual beta2, with EHWS's own CE+KD objective kept fully intact (not an attempt to reproduce ELSA's no-KD recipe). lr and beta2 both still help on top of the lambda fix, and stack cleanly rather than trading off (unlike every Z-step/damping/KD-temperature ablation in Sessions 4-5, which all showed a WT2-vs-C4 tradeoff pattern instead of a joint win).

**Not yet run**: no checkpoint saved for `lr6e-4-beta095` (no `--save-model` passed, exploratory run). Given this result, worth (a) re-running `lr6e-4-beta095` with `--save-model` to get a checkpoint onto HuggingFace, (b) a finer lr sweep around 6e-4 now that the lambda is fixed (2e-4, 4e-4, 6e-4, 1e-3 — the earlier "scaling is non-monotonic" findings were all measured at the wrong lambda, so that surface needs re-exploring), and (c) the still-outstanding real-ELSA-exact (`alpha_kd=0`) comparison, which the user has explicitly deferred in favor of keeping KD active — worth revisiting only if asked.

## Session 6 (2026-09-12): pushed to GitHub, HF upload requested, SLURM batch queue turns out to be a ~1-month dead end, rerun deferred to a fresh interactive session

**Repo pushed to GitHub.** `git@github.com:KimiaGholami/EHWS` had two prior commits (`d1ad7af` initial add, `629fa2e` ELGP→EHWS rename) but none of this project's actual development since -- every file under `ehws/` plus `run_pipeline.py` had 1016+ lines of uncommitted changes sitting locally. Committed and pushed as `0b581f8` (13 files, +2356/-336; `results/` correctly stayed out via `.gitignore`, 1.2GB of experiment outputs excluded). The user-supplied GitHub PAT was used only via a throwaway `GIT_ASKPASS` helper script that reads the token from a short-lived environment variable -- the token never appeared as a command-line argument (this shared node's `ps aux` is visible to other users, see [[slurm-anvil-environment]]), never touched disk, and the remote URL was reverted to its plain form immediately after the push.

**User asked to upload `lr6e-4-beta095`'s weights to `ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct`.** Problem: that exploratory run never passed `--save-model`, so no checkpoint exists -- a full rerun is needed. Logged into HuggingFace with the user-supplied token (same env-var-only pattern, via `huggingface_hub.login()`; persists to `~/.cache/huggingface/token`, a 600-permission file the tool itself manages).

**SLURM batch submission turns out to be a dead end for this.** The interactive OnDemand session's own job (`20549633`) had only ~2h45m of walltime left (45h+ into its 48h hard limit) -- nowhere near the ~11-12h a full HGRN-1.3B Phase1+Phase2 rerun needs. Tried submitting a standalone `sbatch` job instead (fresh 48h walltime, independent of the interactive session) to the `gpu` partition/QOS this account uses: `sbatch --test-only` reported an estimated start of **2026-10-14**, over a month out -- the partition had 429 pending jobs queued ahead of it. No account-level job/TRES limit was the cause (`sacctmgr show assoc` showed none for this account); it's just real, heavy multi-tenant contention on Anvil's shared GPU partition. Standalone batch submission is not a viable path for anything that needs a GPU soon on this cluster -- only the interactive OnDemand allocation mechanism gets one promptly, for whatever reason (dedicated/reserved capacity, different scheduling path, or the user's existing allocation).

**Asked the user how to proceed; they chose to start a fresh OnDemand session** (fresh ~48h walltime) rather than risk the current session's ~2h45m (Phase 1 alone needs ~7.5h, so launching on the old session would almost certainly waste GPU time dying mid-Phase-1 with nothing saved) or look for another cluster path.

**Everything needed is staged, waiting on that fresh session:**
- `results/_lambdafix_batch/launch_savemodel.sh` -- reruns the exact `lr6e-4-beta095` config (`--p2-lr 6e-4 --p2-lambda-max 0.01 --p2-lambda-schedule constant --adam-beta2 0.95`, KD active) with `--save-model` on GPU 0, and automatically fires the auto-upload the moment the training process exits (success or failure).
- `results/_lambdafix_batch/auto_upload.py` -- refuses to push unless (a) `results.json`'s WikiText2/C4 land within 20% of the exploratory run's 85.76/57.21 and (b) an actual `.safetensors` file exists in `model-s0.8/` -- mirrors the gating pattern from the earlier `savemodel-v2` attempt ([[ehws-hgrn-ppl-gap]]).
- `results/_lambdafix_batch/watchdog_savemodel.sh` -- same 5-minute liveness/GPU-memory/ECC/dmesg-Xid-OOM/walltime-remaining monitoring as the 4-way batch's watchdog.
- All three `bash -n`/`py_compile`-verified; none have been run yet. Syntax-clean and ready -- just needs `bash results/_lambdafix_batch/launch_savemodel.sh` once a fresh session with most of its walltime ahead of it is up.

**README.md updated** with the new HGRN-1.3B@80% result (dense 11.84/16.89 → pruned 85.76/57.21 WT2/C4, zeroshot avg 0.435→0.357, vs real ELSA's own 54.08/36.92) and a note in the Hyperparameters section that `ehws/hparams.py`'s HGRN@80% entry is now measured ground truth, not an approximation.

## Session 7 (2026-09-14/15): save-model rerun launched, died at a real `/home` quota wall (not GPU memory), root-caused and fixed, checkpoint uploaded

**Fresh OnDemand session up (job `20617103`, ~1d18h walltime at launch), `launch_savemodel.sh` run.** Training completed cleanly overnight: `results.json` showed WT2 85.7578/C4 57.2065, essentially exact reproduction of the exploratory run's 85.76/57.21 -- confirms the pipeline is effectively deterministic given fixed seeds. But `--save-model` died silently right after zero-shot eval: `run.log` cut off mid-line with zero traceback, `model-s0.8/` created but empty, auto_upload.py correctly refused to push (no `.safetensors`).

**Root cause: `/home` has a hard 25GB per-user quota, and it was at 99.9% full.** `df -h` on `/home` is misleading -- it reports the whole shared filesystem's free space (8.7TB), not this account's actual quota. Neither `quota -s` nor `lfs quota` work on this ZFS-backed mount; the real tool is RCAC's own `myquota` (`/usr/local/bin/myquota`). A relaunch attempt before this was discovered crashed immediately with an explicit `OSError: [Errno 122] Disk quota exceeded`, confirming the diagnosis -- the original silent death was almost certainly the same wall, just hit inside `safetensors.torch.save_file()`'s write without a clean traceback.

**Fix applied:**
1. Deleted the old superseded HGRN pruned checkpoint (`results/hgrn-1.3b-dualphase-0.8-lr3x-ncalib2048-savemodel-v2/model-s0.8/model.safetensors`, the obsolete 422/162-PPL config's weights) per explicit user permission.
2. Redirected `results/_lambdafix_batch/{launch_savemodel.sh,auto_upload.py,watchdog_savemodel.sh}`'s output from `results/...` (on `/home`) to `/anvil/scratch/x-kgholami/ehws-savemodel-out/...` (100TB quota, ~0% used) -- same basename preserved so `run_pipeline.py`'s checkpoint_dir logic still resolves correctly.
3. Hardened `run_pipeline.py`'s save-model block (~line 377-394): moves `model.to("cpu")` before building the state_dict (back to `device` after), added `flush=True` on every print in that block, so a future failure here leaves a real trace instead of dying silently.

**Relaunched, reusing the existing Phase-1 checkpoint (`/anvil/scratch/.../ehws-checkpoints/hgrn-1.3b-lambdafix-lr6e-4-beta095-savemodel/phase1_checkpoint.pt`, 11.5GB) to skip the ~7.5h Phase 1.** Finished 2026-09-15 01:35 with WT2 88.23/C4 56.79 (slightly different from the earlier near-exact match -- Phase 2 was rerun fresh, not resumed, so some bf16 non-determinism crept in; still well within the 20% upload gate). **Checkpoint uploaded successfully** to `ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct` (commit `e9e92525e7`, 2026-09-15 15:34 UTC). Verified via direct blob-hash comparison against the repo's prior (2026-09-09, old 422/162-PPL config) commit: only `model.safetensors` actually changed content; `config.json`/`generation_config.json`/tokenizer files are byte-identical between the two uploads (pruning doesn't touch architecture/tokenizer, so HF correctly keeps their "last modified" attribution on the older commit -- this confused the user briefly, resolved by checking real blob IDs with full (not truncated) commit SHAs).

**Lesson generalized to [[slurm-anvil-environment]]**: for any job writing more than a few hundred MB, check `myquota` before launching and default large-artifact output to `/anvil/scratch/<user>/...`, never `/home` -- `df -h` free space is not evidence of quota headroom on this cluster. Past unexplained silent-death incidents in this project's history may share this same root cause.

## Session 8 (2026-09-15/16): does the two-phase architecture even help? Tested dropping Phase 1 -- result: no, it was actively hurting

**User's question, well-motivated by the numbers**: Phase 1 is actually the *longer* phase (~7.5h solo vs Phase 2's ~4.1h), and total two-phase runtime (~11.6h) is longer than a single continuous run of Phase 2's own step count would need -- because Phase 2's step budget (128 rounds x 32 x_steps = 4096) was never reduced to account for a warm start; it's identical to real ELSA's *entire* single-phase step budget. So Phase 1 is architecturally pure addition on top of a Phase 2 that already does 100% of ELSA's job, not a substitution that makes Phase 2 cheaper. This made "does Phase 1 pull its weight" worth re-testing now that lambda is fixed.

**`--skip-phase1` (via `build_dense_states`, real ELSA's own single-continuous-run architecture) already existed as a flag** and had already been tested once, pre-lambda-fix: `results/hgrn-1.3b-phase2coldstart-0.8/` (2026-09-01) got 594.0/286.8, worse than the era's two-phase baseline -- but fully confounded by the same 200x-too-small lambda bug as everything else from that era, so it wasn't real evidence either way for the current regime.

**Relaunched properly** (`results/_skipphase1_ablation/`, current winning Phase-2 hparams: lr=6e-4, lambda=0.01 constant, beta2=0.95, n_calib=2048, output on scratch from the start this time). **Result (finished 2026-09-15 21:51): WT2 57.10 / C4 41.39** -- dramatically better than the two-phase result (85.76-88.23/56.79-57.21), landing within ~1.06x/~1.12x of real ELSA's own target (54.08/36.92), the closest this whole project has ever gotten. **This flips the pre-fix finding: once the real lever (lambda) is corrected, the two-phase architecture itself looks actively harmful, not helpful** -- Phase 1's per-layer local warm-start was adding ~7.5h of runtime while making the final result meaningfully worse, not better.

**Timing nuance, not fully explained**: total wall-clock was 6h47m -- faster than a full two-phase run (~11.6h) but far slower than the naive ~4.1h estimate (Phase 2 alone, historically). `run.log` shows `run_phase2()` itself only took 2h39m this time (9550.7s, even faster than the ~4.1h historical figure) -- so the "missing" ~4h is somewhere in the unlogged `build_dense_states()` Hessian-computation step or calibration loading, neither of which has a timing print. Confirmed via `py-spy` (twice, 10s apart, on different lines) that the process was genuinely still computing inside `_x_step`/`run_phase2`'s backward pass throughout the long quiet stretch, not stuck.

**Confirmed both variants are semi-structured, ELSA's HGRN comparison run is unstructured.** `ehws/diagonal_projection.py`'s projection (`diagonal_project`/`magnitude_project`) keeps exactly `k_keep` nonzeros per row via `torch.topk(..., dim=1)`, uniform across rows -- semi-structured by construction, unaffected by whether Phase 1 ran (both paths feed the same Z-step). Real ELSA's own run.log for HGRN explicitly logs `sparsity_type = unstructured` -- confirmed directly, not assumed. So even after adopting one-phase, semi-structured sparsity remains the one dimension distinguishing this project's approach from ELSA's, in addition to timing/quality.

**Open question sent to the user's advisor (not yet answered)**: keep developing two-phase as its own methodological contribution despite it currently being both slower and worse, or adopt one-phase as the new baseline and lean on semi-structured sparsity as the main differentiator from ELSA.

**`--save-model` rerun of the one-phase config launched 2026-09-16 01:55** (`results/_1phase_savemodel/`, pid varies -- check `launch.log`), targeting a **new** HF repo `ikimyaii/HGRN-1.3-semi-structured_EHWS-1phase-80pct` (created empty just before launch). No Phase-1 checkpoint to resume from here (skip-phase1 has none), so this redoes the full ~6.8h run from scratch. **Launched with only ~7h10m walltime remaining on the current session -- user was explicitly warned this leaves almost no margin and this run has no resume checkpoint if killed, and chose to proceed anyway rather than start a fresh session.** Output on scratch from the start (lesson from Session 7 already applied). Auto-upload gate: within 20% of 57.10/41.39. Watchdog (`results/_1phase_savemodel/watchdog.sh`) escalates its walltime warning threshold more aggressively than prior watchdogs given the tight margin. **As of this write-up, just launched -- check `results/_1phase_savemodel/` and the new HF repo for the real outcome. If it dies to walltime, it must be relaunched in full, not resumed.**

## Session 9 (2026-09-21/23): OPT-125M testbed -- HGRN-findings audit, a real calibration-loading bug found and fixed, step-cut sweep re-confirms one-phase beats two-phase, user chooses to keep two-phase anyway (differentiation from ELSA), over-relaxed ADMM tried and rejected with a verified (not buggy) negative result

**Housekeeping first**: a model card (`README.md`) was added to the previously-cardless `ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct` HF repo (the dual-phase lambdafix-lr6e-4-beta095 checkpoint, WT2 88.23/C4 56.79, uploaded end of Session 7) -- numbers pulled directly from that exact run's `results.json` on scratch, not the near-but-not-identical exploratory-run figures quoted elsewhere in this file. One drafting error caught before it shipped: an early wording implied real ELSA's paper published an 80%-sparsity HGRN number -- it doesn't (HGRN isn't one of ELSA's own paper models, see Session 3's docstring note in `hparams.py`); corrected to say plainly that *this project* ran ELSA's official code against HGRN itself to get that comparison number, and the card was re-uploaded.

### Auditing whether HGRN's Session-5-8 fixes needed porting to OPT-125M -- mostly already applied, one real documentation error found and corrected

User asked to verify the HGRN lambda-fix/beta2/prox-after-clip findings against OPT-125M before running anything new on it. Checked `ehws/hparams.py` and `ehws/admm.py` directly rather than assuming:

- `prox_after_clip=True` is already the global default (`ehws/admm.py`'s `Phase2Config`) -- already applies to every OPT-125M run, nothing to port.
- The "lambda 200x too small" bug (Session 5) **never existed for OPT-125M.** `_OPT_125M`'s table already used `"constant"` schedule with ELSA's own real published per-sparsity lambda values (transcribed from ELSA's Table 5 back in Session 2, verified cell-by-cell at the time) -- the buggy cosine/5e-5 fallback only ever affected `_HGRN_1_3B`'s untabulated sparsities, which fall through to the (different, unrelated) OPT-1.3B approximation table.
- **Real correction to this file's own Session 4 text**: line ~1108's claim that the OPT-125M `beta2=0.95` ablation (`results/opt-125m-fixablation-beta2-0.95`) ran "with the still-wrong lambda for that run" is **not supported by that run's actual saved config** -- checked `results.json` directly: `no_auto_hparams: false`, meaning auto-hparams *was* on and supplied the correct table values (`lr=6e-4, lambda=1e-3, constant`) for that run, same as its sibling `opt-125m-fixablation-prox-after-clip`. So the beta2=0.95 result (WT2 93.22/C4 54.90) is a clean, non-confounded comparison against `beta2=0.999` (WT2 85.98/C4 52.68, same other settings) -- **beta2=0.95 is a genuine loss on OPT-125M**, not an artifact of a stale lambda. This is architecture-specific and doesn't transfer from HGRN (where beta2=0.95 helped): kept the default (`0.999`) for every OPT-125M run this session rather than porting HGRN's value over.
- `n_calib=2048` and the 3x-lr KD-dilution compensation baked into `_OPT_125M`'s table were both discovered *on* OPT-125M originally (Session 2) -- already there, nothing to port.

### Real bug found and fixed: redundant, CPU-contended calibration/eval data loading across parallel sweep arms

First attempt at a 4-arm parallel sweep (see below) stalled for 1h45m+ with 0% GPU utilization across all 4 processes before being killed. `py-spy dump` on the stuck PIDs confirmed all four genuinely inside `get_c4_calibration`'s reservoir-scan loop (`ehws/calibration.py:50`), not deadlocked -- just very slow. Root cause: that function's pool size is `max(2000, n_samples*20)` -- at `--n-calib 2048` that's **up to 40,960 C4 documents**, fetched from a network-streamed HF dataset and tokenized one-at-a-time, single-threaded, no batching. All 4 arms were doing this identically and redundantly (same model/seed/n_calib -> same calibration set), simultaneously contending for this node's single CPU core (`nproc`=1, same constraint documented for HGRN in Session 3).

**Fix**: added `cached()` to `ehws/calibration.py` (load a tokenized tensor from disk if present, else compute once and atomically write it) and a `--calib-cache <dir>` flag to `run_pipeline.py`, opt-in and `None` by default so existing behavior/results are unaffected. Precomputed the shared calibration/eval tensors once, single process, no contention: **2330.8s (~39min)** for the `n_calib=2048` calibration set (vs. never finishing after 1h45m at 4x contention), 133.2s for the C4 eval set, 1.6s for WikiText-2 (not streamed, unaffected). All sweep arms below point `--calib-cache` at this same shared directory (`/anvil/scratch/x-kgholami/ehws-calib-cache`) and load in seconds instead of re-scanning.

### Step-cut sweep: does warm-starting from Phase 1 let Phase 2 use fewer rounds without losing quality? No -- and one-phase re-confirms it beats two-phase on both axes, same as HGRN

Motivation: Phase 2's round budget (128 rounds = 4096 steps) is never reduced to account for Phase 1's warm start -- it's identical to real ELSA's own full single-phase step budget, so two-phase currently does 100% of ELSA's own work in Phase 2 alone, on top of Phase 1. Tested whether a warm start lets Phase 2 converge in fewer rounds. 4 arms, `facebook/opt-125m` @ 80% sparsity, `--n-calib 2048 --calib-cache ...`, auto-hparams (`lr=6e-4, lambda=1e-3, constant`), all 4 GPUs, watchdog clean throughout (0 warnings):

| Arm | Phase 1 | Phase 2 | Total wall-clock | WT2 | C4 | Zeroshot avg |
|---|---|---|---|---|---|---|
| Two-phase, 128 rounds (historical baseline, `opt-125m-fixablation-prox-after-clip`, no calib-cache) | ~2h34m | ~1h27m (extrapolated) | ~4h22m (extrapolated) | 85.98 | 52.68 | 0.351 |
| **skip-phase1, 128 rounds** | -- | 5258.5s | 3h16.5m | **80.69** | **50.21** | **0.364** |
| two-phase, 64 rounds | 9242.5s | 2576.7s | 3h33m | 98.82 | 57.85 | 0.340 |
| two-phase, 32 rounds | 9237.2s | 1319.5s | 3h16m | 129.49 | 70.78 | 0.343 |
| two-phase, 16 rounds | 9245.3s | 662.0s | 3h9m | 236.28 | 109.43 | 0.326 |

Phase 2's cost scales cleanly linearly with round count (each halving of rounds roughly halves Phase 2's wall-clock: 2576.7 -> 1319.5 -> 662.0s), confirming no hidden per-round overhead. But **Phase 1's ~2h34m is fixed regardless of Phase 2's round count**, so cutting Phase 2 8x (128->16 rounds) only cuts total wall-clock by ~28% while quality collapses (236 vs 86 WT2) -- step-cutting Phase 2 while keeping Phase 1 is a strictly dominated lever, not a real tradeoff to exploit.

**`skip-phase1` (one-phase) beats every two-phase variant on both PPL and time simultaneously** -- exactly the Session 8 HGRN finding, now replicated on a second architecture. C4 50.21 is within ~6% of real ELSA's own published OPT-125M@80% number (47.45, ELSA.pdf Table 5) -- the closest this project has gotten on OPT-125M.

### User's explicit decision: keep two-phase anyway, to differentiate the method from ELSA

User: "I want to keep the two phase because I want to distant my method from ELSA as much as possible." Explicitly flagged to the user before proceeding: the data above means two-phase isn't just slower than one-phase right now, it's also *worse* in PPL on both OPT-125M and HGRN -- so "distance from ELSA" via architecture alone doesn't yet come with a result that stands on its own merits, but this is a legitimate research-positioning call, not something to keep arguing against. Redirected effort toward finding a lever that makes two-phase *itself* faster or gives it a genuine algorithmic point of departure from ELSA, rather than dropping Phase 1.

### Over-relaxed ADMM: implemented, tested, rejected -- a real, verified negative result

New folder `relaxed_admm/` (inside `semi-structured/`, sibling to `ehws/`), keeping the parent `ehws` package's Phase 1, Z-step, and loss untouched and only forking Phase 2's update rule:

- `relaxed_admm/admm_relaxed.py`: `RelaxedPhase2Config` (`Phase2Config` + `relax_alpha: float = 1.0`) and `run_phase2_relaxed`, implementing Boyd et al. 2011 (section 3.4.3) over-relaxation: `x_hat = alpha*x_new + (1-alpha)*z_prev` computed each round *before* the z/u update, with the z-step and u-step then using `x_hat` in place of the raw x-step output (`z = zstep_fn(x_hat + u, H, k_keep)`, `u = u + x_hat - z`). The x-step itself, and what seeds the *next* round's x-step, both stay unrelaxed (`s.x` = raw `x_new`), matching Boyd's formulation exactly -- `x_hat` only ever affects the z/u bookkeeping. `alpha=1.0` reduces to `x_hat == x_new`, i.e. exactly the unmodified baseline.
- `relaxed_admm/run_pipeline.py`: copy of `../run_pipeline.py` wired to the relaxed Phase 2 (`--relax-alpha`, default 1.0), inserts `../` onto `sys.path` so `import ehws` resolves regardless of invocation cwd.
- **Regression check** (smoke test, then a real run): `--relax-alpha 1.0` reproduced the unmodified baseline's per-round ADMM residual and final PPL bit-for-bit on a smoke test, and closely on a real run (84.79/52.55 vs. the historical 85.98/52.68 -- small gap is ordinary run-to-run noise, not the fork).

**Sweep** (same testbed as the step-cut sweep, `--calib-cache` reused, watchdog clean, 0 warnings across the full ~4h run):

| Arm | alpha | Rounds | WT2 | C4 | Zeroshot avg | Phase 2 time |
|---|---|---|---|---|---|---|
| Regression check | 1.0 | 128 | 84.79 | 52.55 | 0.360 | 5243.4s |
| Over-relaxed | 1.5 | 128 | 105.65 | 62.06 | 0.362 | 5242.6s |
| Over-relaxed | 1.8 | 128 | 221.46 | 105.13 | 0.357 | 5241.7s |
| Over-relaxed, half budget | 1.5 | 64 | 133.00 | 73.02 | 0.355 | 2635.3s |

**Over-relaxation hurts, monotonically, at every alpha tested -- and doesn't buy convergence speed either** (alpha=1.5 @ 64 rounds, 133.00, is worse than the *plain*-ADMM 64-round arm from the step-cut sweep, 98.82). Verified this is real, not a bug, by checking the raw per-round ADMM dual residual (`max ||u||/||x0||`) rather than trusting final PPL alone:

| Round | alpha=1.0 | alpha=1.5 | alpha=1.8 |
|---|---|---|---|
| 1 | 2.010 | 1.855 | 1.792 |
| 20 | 0.459 | 0.588 | 0.611 |
| 40 | 0.423 | 0.501 | 0.699 (rose from round 20) |
| 80 | 0.403 | 0.447 | 0.513 |
| 128 | 0.458 | 0.470 | 0.588 |

Two things this confirms: (1) `relax_alpha` is genuinely wired through and active -- the three runs diverge from each other starting at round 1 (2.010/1.855/1.792), which couldn't happen if the flag weren't affecting anything, since seed/data/hyperparameters are otherwise identical; (2) alpha=1.0 converges smoothly and monotonically while alpha=1.8 visibly oscillates (residual *rises* round 20->40 before partially recovering) -- a real instability signature, and the final residual tracks final PPL exactly (0.458/0.470/0.588 vs. 84.79/105.65/221.46). Mechanistically sound, not just correlated: Boyd's proven convergence speedup assumes a well-behaved (typically convex/continuous) proximal operator on the z-side; Phase 2's Z-step is a hard, discrete top-k projection, and extrapolating the iterate fed into it (what `x_hat` does at alpha>1) can flip which specific weights survive from round to round -- a much rougher operator to relax against than what the technique was designed for.

**Decision: over-relaxed ADMM is closed out, not worth retrying at smaller alpha** -- the degradation is monotonic starting from alpha=1.0 itself (1.0 is already the best of the four), so there's no reason to expect an untested value between 1.0 and 1.5 to reverse the trend.

### Not yet tried: adaptive/residual-balancing penalty (proposed, not yet run)

Floated as the next candidate for a genuine two-phase-preserving, ELSA-differentiating, Phase-2-only mathematical change: classic ADMM residual-balancing adjusts the penalty parameter (`admm_lambda`, this project's rho) each round based on the primal/dual residual ratio, instead of the current fixed schedule (constant or cosine-ramped). Doesn't touch the Z-step's hard projection the way over-relaxation did, so it may not hit the same failure mode. Not implemented or tested yet -- next natural candidate if this direction is pursued further.

## Session 10 (2026-09-23): Phase 1 -> Phase 2 handoff fixes and faster Hessians (implemented, verified on a toy model, not yet run on GPU)

Goal: keep two-phase (user's decision, Session 9) but make it cheaper and stop Phase 1 from hurting Phase 2. Two changes, both opt-in or exact, so existing configs reproduce unchanged.

### 1. Handoff: Phase 1's dual is the wrong size for Phase 2

- **Phase 1 still had the "λ 200x too small" setting.** `--p1-lambda` defaults to 5e-5 with a cosine ramp, the same value fixed for Phase 2 in Session 5 but never revisited for Phase 1. With almost no penalty, `u = Σ(x - z)` just accumulates the pruned mass every round. On a toy OPT model, the default Phase 1 hands Phase 2 a dual with `max ||u||/||x0|| = 2.02`, which matches the 2.01 logged at Phase 2 round 1 on real OPT-125M (Session 9 relaxed-ADMM table).
- **`u` is the scaled dual** (the real dual is `λ·u`, and the proximal target is `z - u`), but it was handed over without rescaling. Moving from λ=5e-5 to 1e-3 (OPT-125M) or 1e-2 (HGRN) multiplies the real dual by 20x or 200x at the handoff. Likely explains both "two-phase is worse than one-phase" and "Phase 2 can't be shortened after the warm start" (Session 9 step-cut sweep). Still a hypothesis, not a measured result.
- New flags:
  - `--p2-dual-init {carry,rescale,zero}` (default `carry` = old behavior). `rescale` multiplies `u` by `λ_p1/λ_p2`; `zero` keeps Phase 1's x/z and starts a fresh dual. `LayerState.admm_lambda` records the penalty `u` was built under (saved in the checkpoint; old checkpoints load it as 0, which leaves `rescale` as a no-op).
  - `--p1-lambda-schedule {cosine,constant}` (default `cosine` = old behavior).
- Not applied to `relaxed_admm/`: `run_phase2_relaxed` builds its own Phase 2 state.

### 2. Hessians: cheaper and smaller, same results

- `diagonal`/`magnitude` only ever read `diag(H)`. `LayerHessian(diag_only=True)` accumulates just that vector. It's now automatic for those two Z-steps (`--full-hessian` turns it off). Phase 2 no longer holds every layer's full d_in×d_in H on the GPU (≈11GB on HGRN-1.3B, now a few MB).
- `_compute_hessians`: hooks many layers in one pass, and stops each batch's forward pass as soon as every hooked layer has seen its input (skipping later blocks and the lm_head). Exact.
  - `build_dense_states` (`--skip-phase1`): one pass for all layers instead of one full pass per layer (168 passes → 1 on HGRN). This is almost certainly the unexplained ~4h found in Session 8's one-phase run.
  - `run_phase1` / checkpoint reload: still one layer at a time (each layer must be measured after the previous one is pruned), but now with early exit.
- `--hessian-n-calib N` (default: all, the old behavior) measures H on the first N calibration sequences. This one *is* an approximation. At `--n-calib 2048`, the per-layer Hessian passes were roughly half of Phase 1's compute.
- **Verification** (random 3-layer OPT, CPU, original code vs new code in separate processes): Phase 1 x/z/u and Phase 2 z were bit-identical, both with full and diagonal-only H. Cold-start H matched exactly; decoder-block forward calls for the cold-start pass dropped from 216 to 12. `dual_init` scaled the initial dual exactly (1.0 / 0.1 / 0). Checkpoint round trip matched.
- The checkpoint `meta` now also records Phase 1's lambda/schedule/lr/rounds/x_steps/zstep and `hessian_n_calib`, so older `phase1_checkpoint.pt` files will be rejected (they're normally deleted on success anyway).

### Fixed: Phase 1's trained biases weren't saved in the checkpoint

`_x_step` optimizes each layer's `bias` along with its `weight` (confirmed: all 18/18 toy-OPT layers' biases change in Phase 1). `save_phase1_checkpoint` saved only x/z/u, so a run resumed from a checkpoint started Phase 2, and recomputed every later layer's H, with the dense model's biases. That affected any resumed OPT run (OPT linears have biases). **Fix:** the checkpoint now stores each layer's trained bias, and `load_phase1_checkpoint` restores it during the H replay (after that layer's own H, matching the order `run_phase1` measured in). Checkpoints without a `bias` key keep the old behavior. **Verified** on the toy model: a resumed run gives the same biases, the same H, and the same Phase 2 z and final biases as a run that never stopped.

**Keep training biases -- real ELSA does too, and more.** `ELSA-official/lib/trainer.py::create_optimizer` (checked at github.com/log-postech/elsa `03322d6`) puts every parameter with `requires_grad` into the optimizer. The `nn.Linear` weights (except lm_head) form the ADMM group; everything else goes in an `other_params` group that gets plain Adam with no proximal term. Nothing in `main.py`/`lib/utils.py` freezes anything, so ELSA also trains **LayerNorms, embeddings, and the lm_head**. EHWS trains only the prunable linears' weights and biases (everything else is frozen in `run_pipeline.py`). That's a real, untested difference from ELSA, and a possible quality lever for Phase 2 (costs extra Adam memory for embeddings/lm_head). Not changed.

### NEXT GPU SESSION: runbook for testing Session 10 (OPT-125M @80%, 4 GPUs, ~4-5h)

**Goal**: find out whether (a) `--hessian-n-calib 256` makes Phase 1 faster without hurting quality, and (b) fixing the Phase 1 -> Phase 2 handoff makes two-phase better, and then whether it lets Phase 2 run fewer rounds (the way two-phase could match ELSA's time).

**Reference numbers (Session 9, same model/sparsity/calibration; WT2 / C4 PPL, lower is better):**
- Two-phase, 128 rounds: **84.79 / 52.55** (Phase 1 ~9240s, Phase 2 ~5240s)
- One-phase (`--skip-phase1`), 128 rounds: **80.69 / 50.21**
- Two-phase, 64 rounds: 98.82 / 57.85; 32 rounds: 129.49 / 70.78

**Before launching:**
- Pull the latest code (`git pull` in `EHWS/`). Session 10 changes are in `ehws/admm.py`, `ehws/hessian.py`, `ehws/diagonal_projection.py`, `run_pipeline.py`.
- Start a fresh OnDemand session with most of its 48h walltime left (see Session 3/5 silent-death notes).
- The calibration cache from Session 9 (`/anvil/scratch/x-kgholami/ehws-calib-cache`) should already hold the n=2048 set. If it's missing, build it once in a single process first (~39min) rather than letting 4 arms build it at the same time on the 1 CPU core.
- Old `phase1_checkpoint.pt` files will be rejected (the checkpoint meta now records Phase 1's settings). That's expected, not an error; delete them or use new `--out` names (below).
- Use `python3 -u` so `run.log` isn't block-buffered (Session 4 finding) and the `Phase 1 complete in ...s` / `Phase 2 ... complete in ...s` timing lines show up live.

**Launch (one arm per GPU):**
```
cd /home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured
PY=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3
COMMON="--model facebook/opt-125m --sparsities 0.8 --zeroshot-sparsities 0.8 --n-calib 2048 \
  --calib-cache /anvil/scratch/x-kgholami/ehws-calib-cache --hessian-n-calib 256"
launch() {  # $1 = GPU, $2 = run name, rest = extra flags
  gpu=$1; name=$2; shift 2; mkdir -p results/$name
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY -u run_pipeline.py $COMMON "$@" --out results/$name \
    > results/$name/run.log 2>&1 < /dev/null & disown
}
launch 0 opt-125m-s10-hess256
launch 1 opt-125m-s10-dualzero    --p2-dual-init zero
launch 2 opt-125m-s10-dualrescale --p2-dual-init rescale
launch 3 opt-125m-s10-p1lam-const --p1-lambda 1e-3 --p1-lambda-schedule constant --p2-dual-init rescale
```
(Auto-hparams supplies Phase 2's `lr=6e-4, lambda=1e-3, constant`, same as Session 9. That's why arm 3 sets Phase 1's lambda to 1e-3: both phases then use the same penalty.)

**What each arm answers:**

| GPU | Run | Question | Compare to |
|---|---|---|---|
| 0 | `s10-hess256` | Does Hessian subsampling keep quality? How much faster is Phase 1? | Two-phase 84.79/52.55 and Phase 1 ~9240s |
| 1 | `s10-dualzero` | Is the oversized inherited dual what hurts two-phase? | Arm 0 (same Hessian setting, so it's the fair baseline) |
| 2 | `s10-dualrescale` | Same, but keep the dual's direction at the correct size | Arm 0 |
| 3 | `s10-p1lam-const` | Does Phase 1 with a real penalty give a useful warm start? | Arm 0 |

**Check progress / results:**
```
pgrep -af "run_pipeline.py"
for d in results/opt-125m-s10-*; do echo "-- $d"; grep -E "Phase 1 complete|Phase 2 .*complete|achieved_sparsity|Traceback|Error" $d/run.log; done
```
Also check that Phase 2's round-1 `max ||u||/||x0||` in `run.log` has dropped from ~2.0 (Session 9) in arms 1-3. That's the direct sign the handoff fix is doing what it should.

**How to decide:**
1. **Arm 0 vs 84.79/52.55:** if quality holds (within ~2-3%), keep `--hessian-n-calib 256` for all future runs and record the Phase 1 time saved. If it's clearly worse, try 512.
2. **Arms 1-3 vs arm 0:** if one wins on both WT2 and C4, make that setting the default in code (`Phase2Config.dual_init` / `--p1-lambda-schedule`), the same way `prox_after_clip` was promoted in Session 5.
3. **Winner vs one-phase 80.69/50.21:** if a two-phase arm beats it, two-phase beats ELSA's architecture on quality, not just by being different.
4. **Then the step-cut test** (the real question): rerun the winning arm with `--p2-rounds 64` and `--p2-rounds 32` (2 GPUs). If quality stays near its 128-round number (unlike Session 9's 98.82/129.49 collapse), a good warm start lets Phase 2 be shorter. That's how two-phase can fit in ELSA's time.
5. If no arm beats arm 0, the handoff isn't the problem. Next candidates: residual-balancing lambda (Session 9), and training LayerNorms/embeddings/lm_head like ELSA (see the bias section above).
6. Once OPT-125M shows a winner, repeat it on HGRN-1.3B (~12h per run, needs `--dtype bfloat16 --gradient-checkpointing --p2-micro-batch 1 --p2-grad-accum 8`, and HGRN's Phase 2 lambda is 1e-2, so `--p1-lambda 1e-2` for the arm-3 setting).

Record every result back in this file.

**Why not run this on a laptop:** timed on an M1 Pro (2026-09-23), one OPT-125M Phase 2 step at seqlen 2048 / batch 8 takes 27s on CPU and 212s on MPS (likely memory pressure from full-vocab logits in 16GB), vs ~1.3s on an A100. That's ~31h for Phase 2 alone on CPU, 2-3 days per arm with Phase 1. It needs the GPUs.

## Session 11 (2026-09-24 to 2026-09-26): Session 10 runbook results -- `rescale` handoff wins, made default

Ran the Session 10 runbook as written: OPT-125M @80%, n_calib=2048, calibration cache, `--hessian-n-calib 256` on all four arms, one arm per A100, all four in parallel. All finished cleanly (no crashes, all `results.json` written). Auto-hparams supplied Phase 2's lr=6e-4, lambda=1e-3, constant, as in Session 9.

### Results (WT2 / C4 PPL, lower is better; zero-shot = 7-task average accuracy)

| Arm | Handoff | WT2 | C4 | Zero-shot avg | Avg w/o BoolQ | Phase 2 round-1 `max \|\|u\|\|/\|\|x0\|\|` |
|---|---|---|---|---|---|---|
| `s10-hess256` (arm 0) | carry | 87.04 | 53.77 | 0.369 | 0.327 | 1.87 |
| `s10-dualzero` | zero | 84.11 | 52.63 | 0.375 | 0.334 | 0.66 |
| `s10-p1lam-const` | P1 lambda=1e-3 constant + rescale | 82.03 | 51.80 | 0.366 | 0.323 | 1.68 |
| **`s10-dualrescale`** | **rescale** | **80.89** | **51.55** | 0.340 | 0.325 | 0.69 |
| Session 9 two-phase (full H) | carry | 84.79 | 52.55 | | | ~2.0 |
| Session 9 one-phase | -- | 80.69 | 50.21 | | | |

Timing (all four arms nearly identical): Phase 1 **~4207s** (Session 9: ~9240s), Phase 2 ~5271s (Session 9: ~5240s). All arms reached achieved_sparsity 0.7997.

### What it means

- **Hessian subsampling (arm 0 vs Session 9 two-phase):** Phase 1 is 2.2x faster, for +2.7% WT2 / +2.3% C4. That's right at the runbook's "within ~2-3%" line: keep `--hessian-n-calib 256` for iteration, but use the full H for headline numbers (or try 512).
- **The handoff hypothesis is confirmed.** All three fixes beat arm 0, and `rescale` is best on both WT2 (-7.1%) and C4 (-4.1%). `zero` helps less (-3.4% / -2.1%), so Phase 1's dual carries useful direction and was only the wrong size. `p1lam-const` (same penalty in both phases, so `rescale` is a no-op) lands close to `rescale`, which confirms the same diagnosis from another angle.
- **Dual size converges anyway:** by Phase 2 round 32 every arm (including carry) sits at `max ||u||/||x0||` ~0.45-0.48, ~0.40-0.43 at round 64/96. The oversized dual does its damage in the first rounds, and the model doesn't fully recover from it.
- **Versus one-phase:** `dualrescale` ties it on WT2 (80.89 vs 80.69) but is 2.7% worse on C4 (51.55 vs 50.21). Not a clear win yet, and one-phase used the full H, so that comparison isn't like-for-like.
- **Zero-shot doesn't separate the arms.** `dualrescale`'s lower average comes entirely from BoolQ (0.43 vs ~0.62 in the others). Without BoolQ all four are within ~1 point (0.323-0.334), which is noise for a 125M model near chance. Use PPL to decide at this scale.
- Single seed per arm. The `rescale` vs `p1lam-const` gap (~1.4% WT2 / 0.5% C4) is small enough that a second seed is worth running before drawing conclusions about Phase 1's lambda.

### Code change: `--p2-dual-init rescale` made the default

`Phase2Config.dual_init` and `run_pipeline.py`'s `--p2-dual-init` now default to `rescale` (was `carry`); pass `--p2-dual-init carry` to reproduce older runs. Phase 1's lambda/schedule defaults are **unchanged** (5e-5, cosine). `rescale` scored better than matching lambdas, it's correct under any pair of penalties (a no-op when they're equal), and a matched Phase 1 lambda would need per-model tuning (e.g. 1e-2 for HGRN, untested in Phase 1).

### Next (fresh OnDemand session needed; this one had <3h walltime left when the sweep finished)

1. **Step-cut test** (runbook step 4): `dualrescale` config with `--p2-rounds 64` and `--p2-rounds 32`. Session 9's collapse (98.82 / 129.49 WT2) was measured with the carry bug, so it needs remeasuring. If quality stays near 81, two-phase can run in less time than ELSA.
2. **Like-for-like one-phase comparison:** `dualrescale` without `--hessian-n-calib` (full H), to see whether it beats one-phase's 80.69 / 50.21.
3. Optional: second seed of `dualrescale` and `p1lam-const`.
4. Then port the winner to HGRN-1.3B (runbook step 6).

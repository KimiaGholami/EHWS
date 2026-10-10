# EHWS -- semi-structured ADMM pruning

EHWS stands for Extreme Hierarchical Weight Sparsity. A two-phase pruning method for causal language models that trains
directly against the true next-token loss under a hard sparsity
constraint, instead of matching each layer's dense output on calibration
data and hoping that adds up to a good model. It extends
[ELSA](https://arxiv.org/abs/2510.01650) (*The Unseen Frontier: Pushing
the Limits of LLM Sparsity with Surrogate-Free ADMM*), which makes the
same argument for a single flat optimization; this method splits it into
a cheap per-layer warm start followed by one global fine-tune, so you
only pay for the expensive part once per model instead of once per
sparsity level.

"Semi-structured" here means every row of every prunable linear layer
keeps exactly the same number of weights (e.g. the top 20% at 80%
sparsity). Unstructured pruning, by contrast, lets rows and layers end up
with different amounts pruned. Which weights survive within a row is
still free (no fixed block pattern such as 2:4).

## Method

Both phases are the same ADMM structure: a dense trainable copy of the
weights `x`, a sparse projection of it `z`, and a dual variable `u` that
tracks how far apart they are.

```
x <- argmin_x  f(x) + (lambda/2) ||x - z + u||^2      (train against the true loss)
z <- argmin_{||z||_0<=k} (z-v)^T H (z-v), v = x + u    (project onto the sparsity budget)
u <- u + x - z                                          (track the disagreement)
```

`f` is cross-entropy plus a knowledge-distillation term against the
original dense model (`f = 0.5 * CE + 0.5 * KD`) -- both computed on
unlabeled text, so no task-specific data is needed. `H` is the standard
SparseGPT/CWS reconstruction Hessian (`2/N * sum x x^T` over calibration
activations); the Z-step keeps the `k` weights per row with the largest
`H_ii * v_i^2` score and zeros the rest, which is the exact solution to
the diagonal-approximated version of the projection above.

**Phase 1** prunes each layer on its own, sequentially, in execution
order. The default for the headline results below is the **local
(ALPS-style) Phase 1** (`--p1-mode local`). Each layer minimises its own
reconstruction error `||(W - W0) X||^2` under the same per-row constraint.
It uses ADMM with a closed-form x-step and a penalty grown from 0.01 to
10 x mean(diag H), then an exact least-squares refit of the kept weights.
There's no backprop and no teacher, so it takes about 10 minutes on
HGRN-1.3B. The older global Phase 1 (`--p1-mode global`) trains each layer
against the full-model CE+KD loss instead and costs hours.

**Phase 2** takes Phase 1's pruned model and makes every layer
trainable at once, under one shared forward pass -- so gradients can
couple across layers instead of being optimized one at a time. All
layers share one uniform target sparsity ratio. This gets re-run once per
sparsity level you actually want a model at. Phase 2 can train in float32
master precision under bf16 autocast (`--master-fp32`), on ELSA's own
32,768 tokenized C4 training sequences with each seen once
(`--elsa-train-cache`, `--p2-epoch-sampling`). It can also train the
non-pruned parameters (norms, embeddings, lm_head) with no penalty
(`--p2-train-extra`), and re-measure diag(H) for the Z-step from its own
training forward passes (`--p2-hessian-refresh`).

## Measured result

`fla-hub/hgrn-1.3B-100B`, 80% sparsity. Perplexity uses ELSA-official's
own evaluation protocol (its WikiText-2 / C4 test sets and formula, fp16),
so every row is directly comparable. All rows are single runs.

| | Sparsity type | WikiText2 PPL | C4 PPL |
|---|---|---|---|
| Dense | -- | 11.81 | 16.09 |
| **Two-phase, local (ALPS) Phase 1 (this method)** | **semi-structured** | **46.49** | **34.58** |
| ELSA official code | semi-structured (same per-row constraint) | 58.79 | 40.30 |
| ELSA official code | unstructured | 54.08 | 36.92 |

Zero-shot accuracy (7-task average): dense 0.438, this model 0.377. The
weights are on Hugging Face:
[ikimyaii/HGRN-1.3B-semi-structured-EHWS-ALPS-phase1-80pct](https://huggingface.co/ikimyaii/HGRN-1.3B-semi-structured-EHWS-ALPS-phase1-80pct).

This beats ELSA under the same row-wise constraint by 21% (WikiText2) and
14% (C4), and also beats ELSA's unstructured run. The result came from
four Phase 2 changes on top of the local Phase 1:
- float32 master weights (HGRN previously trained in pure bf16)
- ELSA's 32,768 training sequences, each seen once (previously 2,048,
  resampled)
- training the non-pruned parameters, as ELSA does
- refreshing diag(H) during Phase 2

Earlier versions of the same two-phase pipeline scored 71.41 / 45.87
(local Phase 1 without those changes) and 87.73 / 55.46 (global Phase 1).
The four changes were applied together, so their individual
contributions aren't separated yet. See `PROGRESS.md` (Sessions 14-16).

An earlier audit against ELSA's own run on this model is also why any of
this works: it showed the ADMM penalty (`lambda`) had been 200x too small
for HGRN (`0.01` constant in ELSA's run vs `5e-5` cosine-ramped here).
See `ehws/hparams.py`'s module docstring.

## Running HGRN-1.3B (80% sparsity)

Set up the environment once:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Two-phase with local Phase 1 (46.49 / 34.58 above).** The training
data is ELSA's own tokenized cache. Build it once by running ELSA-official
with `--admm_save_inputs=True` on this model, or point
`--elsa-train-cache` at an existing copy.

```bash
python run_pipeline.py --model fla-hub/hgrn-1.3B-100B \
  --sparsities 0.8 --zeroshot-sparsities 0.8 --seqlen 2048 --n-eval-c4 256 \
  --micro-batch 2 --grad-accum 4 --p2-micro-batch 1 --p2-grad-accum 8 \
  --dtype bfloat16 --master-fp32 --gradient-checkpointing \
  --no-auto-hparams --p2-lr 6e-4 --p2-lambda-max 0.01 --p2-lambda-schedule constant --adam-beta2 0.95 \
  --p1-mode local --hessian-n-calib 256 \
  --elsa-train-cache ELSA-official/dataset/fla-hub_hgrn-1.3B-100B/train/c4_32768_2048_hub \
  --p2-epoch-sampling --p2-train-extra --p2-hessian-refresh \
  --out results/hgrn-1.3b-0.8
```

**Every non-default flag above is load-bearing, not cosmetic:**

- `--no-auto-hparams --p2-lr 6e-4 --p2-lambda-max 0.01 --p2-lambda-schedule constant --adam-beta2 0.95` --
  without this, `run_pipeline.py` falls back to `ehws/hparams.py`'s auto-hparams
  table, which for HGRN@80% uses `lr=2e-4` (ELSA's raw, un-retuned value) and
  the stock Adam `beta2=0.999`. That combination was actually run
  (`results/hgrn-1.3b-lambdafix-lr2e-4/`) and scores **WT2 207.7 / C4 125.2**
  -- 2-3x worse, with no warning or error. `lambda=0.01` is already correct
  in the auto-hparams table; `lr` and `beta2` are not.
- `--dtype bfloat16 --master-fp32` -- the teacher is stored in bf16. The
  student keeps float32 weights, gradients and Adam state and runs its
  forward under bf16 autocast, as ELSA does. Without `--master-fp32`, the
  bf16 weights are trained directly, and small late-schedule updates
  round away.
- `--gradient-checkpointing --p2-micro-batch 1 --p2-grad-accum 8` --
  memory only, no effect on the result. Phase 2 peaks at about 34 GB, so it
  needs an A100-40GB-class GPU or better.

**Runtime and reproducibility:** about 4h wall-clock on one A100-40GB
(local Phase 1 ~10 min, Phase 2 ~3.3h, plus evaluation). Expect
run-to-run variance, not bit-exact reproduction: repeats of identical
configs have moved WikiText2 by up to ~5% (C4 is much more stable). See
`PROGRESS.md`.

## Hyperparameters

Phase 2's `(lr, lambda, schedule)` per model/sparsity come from
`ehws/hparams.py`, sourced from ELSA's own published values for the
models it covers (OPT-125M, OPT-1.3B) -- see that file for the exact
table and how the untabulated points were filled in. `fla-hub/hgrn-1.3B-100B`
at 80% sparsity is the one entry backed by *measured* ground truth rather
than an approximation: ELSA's own code was actually run against this
model, and its logged hyperparameters (`lr=2e-4, lambda=0.01 constant,
beta2=0.95`) replaced an earlier OPT-1.3B-table guess that turned out to
be 200x too small on `lambda` -- every other HGRN sparsity still falls
back to that same approximation.

**The auto-hparams table's `lr=2e-4` is ELSA's raw value, not what this
repo's headline HGRN results use.** Because this project's objective
mixes in a KD term ELSA's doesn't have (`f = 0.5*CE + 0.5*KD` vs ELSA's
pure CE), the effective task-loss gradient is diluted relative to
ELSA's own tuning assumption. The headline HGRN result above uses `lr`
retuned to `6e-4` (3x) to compensate, on top of the corrected `lambda`. `beta2=0.95` is a plain manual override -- it has
no auto-hparams table entry at all and silently defaults to PyTorch's
stock `0.999` if you forget it. See
[Running HGRN-1.3B](#running-hgrn-13b-80-sparsity) for the exact flags
this actually requires, and `ehws/hparams.py`'s module docstring for the
full numeric comparison. The local Phase 1 has no learning rate or
lambda. Its settings are `--p1-local-rounds 50`, `--p1-local-rho-start
0.01`, `--p1-local-rho-end 10` and the refit (on by default). Phase 2
mixes in a knowledge-distillation term at `alpha_kd=0.5`. Full config for any run
is dumped verbatim into `results/<name>/results.json`.

## Calibration & evaluation

Calibration/training data is either `--n-calib` sequences of 2048
tokens sampled from C4's training split, or ELSA's own cached training
set (`--elsa-train-cache`). Perplexity is reported two ways. Our
protocol uses WikiText-2's test split and a held-out slice of C4's
validation split. ELSA's protocol (`wikitext2_ppl_elsa` /
`c4_ppl_elsa`) uses ELSA-official's own test sets, formula and fp16 eval,
and matches its printed numbers exactly. Compare against ELSA only with
the ELSA-protocol numbers. `eval_external.py` rescores any saved state
dict under both protocols and checks that every row has the same number
of nonzeros.
Zero-shot accuracy uses the standard 7-task 0-shot set (ARC-Easy/
Challenge, BoolQ, HellaSwag, OpenBookQA, RTE, Winogrande) via
`lm-evaluation-harness`.

## Architecture support

Any decoder-only causal LM built from `nn.Linear` layers inside repeated
transformer blocks -- covers OPT, LLaMA, and HGRN (`fla-hub/hgrn-1.3B-100B`)
out of the box, via an execution-order trace rather than hardcoded
per-architecture wiring (`ehws/model_layers.py`).

## Layout

```
ehws/
  hessian.py              # H = (2/N) X^T X accumulation
  diagonal_projection.py  # the Z-step: keep top-k by H_ii * v_i^2 per row
  model_layers.py          # architecture-agnostic prunable-layer discovery
  losses.py                 # the X-step's objective: CE + KD
  admm.py                    # Phase 1 / Phase 2 drivers
  calibration.py              # C4 calibration + WikiText2/C4 eval data
  perplexity.py                # perplexity evaluation
  zeroshot.py                   # lm-eval-harness wrapper
  hparams.py                     # per-model/per-sparsity (lr, lambda) lookup
run_pipeline.py                   # end-to-end driver
requirements.txt
```

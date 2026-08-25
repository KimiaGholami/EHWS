# EHWS-U -- unstructured, global-budget ADMM pruning

EHWS stands for Extreme Hierarchical Weight Sparsity. A two-phase pruning method for causal language models that trains
directly against the true next-token loss under a hard sparsity
constraint, instead of matching each layer's dense output on calibration
data and hoping that adds up to a good model. It extends
[ELSA](https://arxiv.org/abs/2510.01650) (*The Unseen Frontier: Pushing
the Limits of LLM Sparsity with Surrogate-Free ADMM*), which makes the
same argument for a single flat optimization; this method splits it into
a cheap per-layer warm start followed by one global fine-tune, so the
expensive part is paid for once per model instead of once per sparsity
level.

This is the unstructured sibling of EHWS (semi-structured): instead of
forcing every layer to the same sparsity ratio, the network gets one
shared, global weight budget, and the data decides how much of it each
layer gets. A layer that turns out to be collectively redundant gives up
proportionally more of the budget; a layer that matters more keeps
proportionally more. Nothing forces any two layers to land at the same
sparsity.

## Method

Both phases are the same ADMM structure: a dense trainable copy of the
weights `x`, a sparse projection of it `z`, and a dual variable `u` that
tracks how far apart they are.

```
x <- argmin_x  f(x) + (lambda/2) ||x - z + u||^2      (train against the true loss)
z <- argmin_{sum_l ||z_l||_0<=k_total} sum_l (z_l-v_l)^T H_l (z_l-v_l)   (one shared budget across every layer)
u <- u + x - z                                          (track the disagreement)
```

`f` is cross-entropy plus a knowledge-distillation term against the
original dense model (`f = 0.5 * CE + 0.5 * KD`), computed on unlabeled
text. `H` is the standard SparseGPT/CWS reconstruction Hessian
(`2/N * sum x x^T` over calibration activations). Each weight's score is
`H_ii * v_i^2` (the cost of forcing that weight to zero), normalized by
its own layer's mean score first -- otherwise a layer's initialization
scale alone (not real importance) can dominate which layers get pruned;
see `ehwsu/global_projection.py` for why this matters and what happens
without it.

**Phase 1** runs this per layer, sequentially: one continuous ADMM run
straight at a fixed per-layer target sparsity, with the penalty smoothly
ramped up over the run (cosine schedule, 0 to its max value) instead of
a discrete ladder. This is a warm start that only depends on the model,
not on any final target, so it runs once.

**Phase 2** takes Phase 1's warm-started state and makes every layer
trainable at once under one shared forward pass, so gradients can couple
across layers. Every round, the Z-step ranks every weight in the network
together and keeps the globally highest-scoring `k_keep_total` of them --
one shared count budget, no per-layer split. This gets re-run once per
overall sparsity level you actually want a model at.

## Measured result

`facebook/opt-125m`, 50% overall sparsity, default settings:

| | WikiText2 PPL | C4 PPL |
|---|---|---|
| Dense | 27.66 | 25.17 |
| ELSA (published, 50%) | 34.14 | 31.52 |
| **This method (50% sparsity)** | **35.05** | **29.80** |

Within 3% of ELSA's own published WikiText2 number, and better than
ELSA's own published C4 number.

Reproduce with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py --model facebook/opt-125m --sparsities 0.5 --out results/opt-125m
```

A single A100-40GB run takes on the order of a few hours for OPT-125M at
this calibration size (n_calib=1024) -- most of it in Phase 2's 4096
optimizer steps and the larger calibration pass the Z-step's Hessian is
built from.

## How this number was reached

Three things mattered, roughly in order of size of effect:

1. **The learning rate for Phase 2 needed retuning for this method's
   objective.** ELSA's own published learning rate was tuned for their
   pure cross-entropy loss; this method's loss also mixes in a
   distillation term, which dilutes the effective gradient magnitude
   enough that ELSA's own value is too conservative here. Scaling it to
   3x closed most of the remaining gap by itself (50.99 -> 35.05
   WikiText2 PPL) -- see `ehwsu/hparams.py`'s docstring. This has only
   been validated at the one (OPT-125M, 50%) point; other sparsity
   levels still use ELSA's unmodified published values.
2. **Calibration set size matters a lot.** Going from 128 to 1024
   calibration sequences was the next-largest single improvement found.
   This tracks a broader pattern in the paper this method is built on:
   training directly against the true loss keeps improving with more
   data, in a way that reconstruction-error-based methods don't.
3. **Phase 1's target sparsity is a real, tunable choice, not a free
   parameter to set by intuition.** 0.7 came from a real sweep over
   {0.3, 0.5, 0.7, 0.9}; 0.9 (the first value tried, chosen by reasoning
   rather than data) was the *worst* of the four -- pushing Phase 1 too
   sparse leaves Phase 2 too little room to recover weights it turns out
   to still need.

## Hyperparameters

Phase 2's `(lr, lambda, schedule)` per model/sparsity come from
`ehwsu/hparams.py` -- see that file for the full table, which points
come from ELSA's own published values, and which one point has been
empirically retuned for this method's objective (see above). Phase 1 has
no published reference (it's this method's own addition):
`target_sparsity=0.7`, 14 rounds of 4 optimizer steps each per layer,
`lr=2e-4`, `admm_lambda_max=5e-5`. Both phases mix in a
knowledge-distillation term at `alpha_kd=0.5`. Full config for any run is
dumped verbatim into `results/<name>/results.json`.

## Calibration & evaluation

1024 sequences x 2048 tokens from C4's training split for calibration --
larger than the 128-sequence convention most one-shot pruning methods
use, because this method keeps improving with more calibration data
rather than saturating early (see above). Perplexity is measured on
WikiText-2's test split and a held-out slice of C4's validation split.
Zero-shot accuracy uses the standard 7-task 0-shot set (ARC-Easy/
Challenge, BoolQ, HellaSwag, OpenBookQA, RTE, Winogrande) via
`lm-evaluation-harness`.

## Architecture support

Any decoder-only causal LM built from `nn.Linear` layers inside repeated
transformer blocks -- covers OPT, LLaMA, and HGRN (`fla-hub/hgrn-1.3B-100B`)
out of the box, via an execution-order trace rather than hardcoded
per-architecture wiring (`ehwsu/model_layers.py`).

## Open items

- The 3x learning-rate retuning above has only been validated at one
  (model, sparsity) point. It's a reasonable starting point for other
  sparsity levels and models, but not a verified one.
- Not yet run on models other than OPT-125M.
- Phase 1's own learning rate (`2e-4`) has a small, real effect --
  `1e-4` measured slightly better (50.27 / 37.75 vs. 50.99 / 37.99 at
  the pre-retuned Phase 2 lr) -- but hasn't been combined with the
  Phase 2 retuning above yet.

## Layout

```
ehwsu/
  hessian.py               # H = (2/N) X^T X accumulation
  diagonal_projection.py   # Phase 1's Z-step: per-layer top-k by H_ii * v_i^2
  global_projection.py     # Phase 2's Z-step: ONE global top-k across every layer
  model_layers.py           # architecture-agnostic prunable-layer discovery
  losses.py                  # the X-step's objective: CE + KD
  admm.py                     # Phase 1 / Phase 2 drivers
  calibration.py               # C4 calibration + WikiText2/C4 eval data
  perplexity.py                 # perplexity evaluation
  zeroshot.py                    # lm-eval-harness wrapper
  hparams.py                      # per-model/per-sparsity (lr, lambda) lookup
run_pipeline.py                     # end-to-end driver
requirements.txt
```

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

"Semi-structured" here means every layer ends up at the same sparsity
ratio (e.g. every layer exactly 50% sparse) -- unlike full unstructured
pruning, where different layers can end up with different amounts
pruned. Within a layer, which specific weights survive is still
unstructured (any weight can go, not a fixed block pattern).

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

**Phase 1** runs this per layer, sequentially, climbing a sparsity ladder
(15%, 30%, ..., 95%) one rung at a time. After each rung, if the dual
residual `||u|| / ||x0||` has grown past a threshold, that layer stops --
a heuristic signal that pushing this layer any sparser is starting to
fight the true loss rather than just following it. This only depends on
the model, not on any target sparsity, so it runs once.

**Phase 2** takes Phase 1's warm-started state and makes every layer
trainable at once, under one shared forward pass -- so gradients can
couple across layers instead of being optimized one at a time. All
layers share one uniform target sparsity ratio. This gets re-run once per
sparsity level you actually want a model at.

## Measured result

`facebook/opt-125m`, 50% sparsity, default settings:

| | WikiText2 PPL | C4 PPL |
|---|---|---|
| Dense | 27.66 | 25.17 |
| **This method (50% sparsity)** | **78.10** | **51.02** |

Zero-shot accuracy (7-task average, lm-evaluation-harness): dense 0.377,
pruned 0.385 -- essentially unchanged, within the noise of a 7-task
average this small.

`fla-hub/hgrn-1.3B-100B`, 80% sparsity, `--p2-lr 6e-4 --adam-beta2 0.95`
(both CE and KD active, `alpha_kd=0.5`):

| | WikiText2 PPL | C4 PPL |
|---|---|---|
| Dense | 11.84 | 16.89 |
| **This method (80% sparsity)** | **85.76** | **57.21** |
| Real [ELSA](https://arxiv.org/abs/2510.01650)'s own published result, same model/sparsity (no KD, pure CE) | 54.08 | 36.92 |

Zero-shot accuracy (7-task average): dense 0.435, pruned 0.357.

This result only exists because of a direct audit against ELSA's own
reference implementation run against this exact model -- its logged
hyperparameters showed this package's ADMM penalty strength (`lambda`)
had been tuned 200x too small for HGRN the entire time (`0.01` constant
in real ELSA's measured run vs `5e-5` cosine-ramped here beforehand);
fixing just that one value cut WikiText2 PPL by ~40% in isolation,
before `lr`/`beta2` were retuned around it. See `ehws/hparams.py`'s
module docstring and `PROGRESS.md` for the full comparison.

Reproduce with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py --model facebook/opt-125m --sparsities 0.5 --out results/opt-125m
```

A single A100-40GB run takes on the order of a few hours for OPT-125M
(most of it in Phase 2's 4096 optimizer steps).

## Known limitation

The saturation check that stops a layer's ladder early measures the ADMM
dual residual, `||u|| / ||x0||`. In practice this ends up correlating
more with how much *weight mass* a rung's cut removes than with whether
the true loss is actually hurt by it -- pretrained weights are close to
Gaussian per layer, and the fraction of weight mass in the smallest-k%
of a Gaussian barely depends on that layer's scale, so the check lands
in a similar range for most layers regardless of how much they actually
matter. We tried replacing it with a direct check on the true loss and
measured it performing *worse*, not better (noisier signal from a small
evaluation batch, and no way to catch many individually-small mistakes
adding up across layers) -- so this simpler, if imperfect, check is what
this repo ships with. This is the honest state of Phase 1's stopping
rule, not a solved problem.

## Hyperparameters

Phase 2's `(lr, lambda, schedule)` per model/sparsity come from
`ehws/hparams.py`, sourced from ELSA's own published values for the
models it covers (OPT-125M, OPT-1.3B) -- see that file for the exact
table and how the untabulated points were filled in. `fla-hub/hgrn-1.3B-100B`
at 80% sparsity is the one entry backed by *measured* ground truth rather
than an approximation: ELSA's own code was actually run against this
model, and its logged hyperparameters (`lr=2e-4, lambda=0.01 constant`)
replaced an earlier OPT-1.3B-table guess that turned out to be 200x too
small on `lambda` -- every other HGRN sparsity still falls back to that
same approximation. Phase 1 has no published reference (it's this
method's own addition): ladder `[0.15, 0.30, 0.45, 0.60, 0.75, 0.90,
0.95]`, 2 rounds per rung, 4 optimizer steps per round, `lr=2e-4`,
`admm_lambda=5e-5`, `saturation_tau=0.15`. Both phases mix in a
knowledge-distillation term at `alpha_kd=0.5`. Full config for any run
is dumped verbatim into `results/<name>/results.json`.

## Calibration & evaluation

128 sequences x 2048 tokens from C4's training split for calibration
(same convention as SparseGPT and ELSA). Perplexity is measured on
WikiText-2's test split and a held-out slice of C4's validation split.
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

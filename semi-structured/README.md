# EHWS: semi-structured pruning of language models with ADMM

EHWS (Extreme Hierarchical Weight Sparsity) removes most of the weights of a
language model, for example 80%, and keeps its quality as high as possible.

It builds on [ELSA](https://arxiv.org/abs/2510.01650). ELSA prunes a model by
training it on its real next-token loss while forcing the weights toward a
sparse pattern, using ADMM. EHWS adds a fast first step that prunes each layer
on its own (Phase 1, based on [ALPS](https://arxiv.org/abs/2406.07831)). The
ELSA-style training then starts from that already-pruned model (Phase 2).

**Semi-structured** means every row of every pruned layer keeps exactly the
same number of weights. At 80% sparsity, each row keeps its top 20%. Which
weights survive within a row is free, so there is no fixed block pattern
like 2:4.

## Results

Perplexity is **WikiText-2 / C4** under ELSA's own evaluation protocol (its
test sets and formula, float16), so every row is directly comparable. Lower is
better. Mamba overflows float16, so every Mamba number (dense and pruned) is
bfloat16. Zero-shot is the average accuracy over 7 tasks (ARC-C, ARC-E, BoolQ,
HellaSwag, OBQA, RTE, Winogrande), higher is better.

The baselines come from ELSA's official code, run on the same models:
- **ELSA row-wise** uses ELSA's code with the same constraint as EHWS (every row keeps the same number of weights).
- **ELSA unstructured** is ELSA's own setting, a looser constraint.

"vs best ELSA" compares EHWS with the better of the two. Unless marked
*tuned*, ELSA uses lr 2e-4 and lambda 1e-2, its HGRN-1.3B setting at 80%.
EHWS uses its HGRN 80% recipe unchanged on every model. All results are
single runs (seed 0). Repeating a run can move WikiText-2 by up to about 5%
and C4 by under 0.5%.

### Headline

| Model | Sparsity | EHWS | Best ELSA | EHWS vs best ELSA |
|---|---|---|---|---|
| HGRN-1.3B | 80% | **46.49 / 34.58** | 53.92 / 36.93 | -14% / -6% |
| HGRN-1.3B | 90% (both tuned) | **81.74 / 47.42** | 115.59 / 60.21 | -29% / -21% |
| Gated DeltaNet-1.3B | 80% | **22.88 / 23.67** | 26.19 / 26.24 | -13% / -10% |
| Gated DeltaNet-340M | 80% | **38.58 / 33.23** | 45.95 / 38.18 | -16% / -13% |
| Mamba-1.3B | 80% | **25.03 / 25.49** | 29.08 / 28.92\* | -14% / -12% |
| OPT-125M | 80% | **66.65 / 49.45** | 75.86 / 53.45 (row-wise) | -12% / -7% |

\* ELSA's own float16 number; a bfloat16 re-score is pending.

EHWS has the lower perplexity in every comparison on HGRN, Gated DeltaNet and
Mamba from 30% to 90% sparsity. Zero-shot accuracy is roughly on par (see
[Zero-shot accuracy](#zero-shot-accuracy)).

### HGRN-1.3B across sparsity

`fla-hub/hgrn-1.3B-100B`; dense 11.81 / 16.09, zero-shot 0.438.

| Sparsity | EHWS | ELSA row-wise | ELSA unstructured | EHWS vs best ELSA | Zero-shot EHWS / ELSA row / ELSA unstr |
|---|---|---|---|---|---|
| 30% | **15.35 / 15.67** | 17.25 / 17.64 | 17.18 / 17.64 | -11% / -11% | 0.415 / -- / 0.423 |
| 50% | **17.48 / 16.92** | 19.97 / 19.72 | 19.58 / 19.59 | -11% / -14% | 0.414 / 0.417 / 0.415 |
| 70% | **27.68 / 23.16** | 33.01 / 28.27 | 32.58 / 27.15 | -15% / -15% | 0.389 / 0.388 / 0.392 |
| 80% | **46.49 / 34.58** | 58.79 / 40.30 | 53.92 / 36.93 | -14% / -6% | 0.377 / 0.369 / 0.368 |
| 90%, untuned | 105.72 / 63.59 | 192.68 / 78.18 | 213.97 / 79.08 | -45% / -19% | 0.360 / 0.360 / 0.362 |
| **90%, both tuned** | **81.74 / 47.42** | 144.16 / 68.91 | 115.59 / 60.21 | -29% / -21% | 0.371 / -- / -- |

### Tuning both methods at 90% (HGRN-1.3B)

90% is far from the 80% setting both methods were tuned on, so both were
re-tuned there.

EHWS, one change at a time from the 80% recipe:

| Change | WikiText-2 / C4 | Zero-shot |
|---|---|---|
| none (Phase 1 straight to 90%) | 105.72 / 63.59 | 0.360 |
| Phase 2 lr 4e-4 (was 6e-4) | 107.67 / 65.02 | 0.356 |
| Phase 1 to 80%, Phase 2 to 90% | 84.22 / 49.66 | 0.356 |
| **Phase 1 to 70%, Phase 2 to 90%** | **81.74 / 47.42** | **0.371** |
| Phase 1 to 60%, Phase 2 to 90% | 86.18 / 48.94 | 0.366 |

ELSA, starting from the settings in the ELSA paper's Table 5 at 90%:

| ELSA variant and setting | Adam beta2 | WikiText-2 / C4 |
|---|---|---|
| unstructured, lambda 1e-2 (original) | 0.95 | 213.97 / 79.08 |
| unstructured, lambda 1e-2 | 0.999 | 281.20 / 96.97 |
| **unstructured, lambda 1e-3** | 0.999 | **115.59 / 60.21** |
| row-wise, lambda 1e-3 | 0.999 | 144.16 / 68.91 |

The lambda carried over from 80% badly under-tunes ELSA at 90%. With both
methods tuned, EHWS is 29% / 21% lower than ELSA unstructured and 43% / 31%
lower than ELSA row-wise.

### Phase 1 to 70% on other models

The best HGRN 90% setting stops Phase 1 at 70% and lets Phase 2 prune the rest
of the way. Applied unchanged to other models:

| Model | Sparsity | Phase 1 straight to target | Phase 1 to 70% | Change |
|---|---|---|---|---|
| HGRN-1.3B | 90% | 105.72 / 63.59 | **81.74 / 47.42** | -23% / -25% |
| Gated DeltaNet-1.3B | 90% | 44.83 / 38.99 | **37.67 / 32.78** | -16% / -16% |

Mamba-1.3B and Gated DeltaNet-340M at 90%, and all models at 80%, are running.

### Gated DeltaNet

`m-a-p/1.3B-100B-GatedDeltaNet-pure` and `m-a-p/340M-20B-GatedDeltaNet-pure-baseline`.

| Model | Sparsity | Dense | EHWS | ELSA row-wise | ELSA unstructured | EHWS vs best ELSA | Zero-shot EHWS / ELSA row / ELSA unstr (dense) |
|---|---|---|---|---|---|---|---|
| 1.3B | 80% | 10.31 / 13.76 | **22.88 / 23.67** | 26.19 / 26.24 | 26.28 / 26.17 | -13% / -10% | 0.399 / 0.408 / 0.408 (0.512) |
| 1.3B | 90% | 10.31 / 13.76 | 44.83 / 38.99 | running | 46.76 / 43.13 | -4% / -10% | 0.369 / -- / -- |
| 1.3B | 90%, Phase 1 to 70% | 10.31 / 13.76 | **37.67 / 32.78** | running | 46.76 / 43.13 | -19% / -24% | 0.381 / -- / -- |
| 340M | 80% | 16.42 / 19.94 | **38.58 / 33.23** | 47.47 / 38.31 | 45.95 / 38.18 | -16% / -13% | 0.380 / 0.380 / 0.376 (0.437) |

At 90% on Gated DeltaNet, ELSA still uses lambda 1e-2. On HGRN, lambda 1e-3
helped ELSA a lot at 90%, so these margins may shrink once ELSA is tuned here.

### Mamba-1.3B

`fla-hub/mamba-1.3B-100B`, all numbers in bfloat16. Both methods keep Mamba's
`dt_proj` dense (about 2% of the block's linear weights).

| Sparsity | Dense | EHWS | ELSA row-wise | ELSA unstructured | EHWS vs best ELSA | Zero-shot EHWS / ELSA unstr (dense) |
|---|---|---|---|---|---|---|
| 80% | 10.03 / 14.12 | **25.03 / 25.49** | 29.08 / 28.92\* | 30.13 / 28.49 | -14% / -12% | 0.380 / 0.384 (0.435) |

\* ELSA's own float16 number; a bfloat16 re-score is pending.

### OPT-125M (80%)

| | WikiText-2 / C4 | Zero-shot |
|---|---|---|
| Dense | 27.61 / 26.52 | -- |
| **EHWS** | **66.65 / 49.45** | 0.366 |
| ELSA row-wise | 75.86 / 53.45 | 0.375 |
| ELSA unstructured (number from the ELSA paper) | 65.30 / 47.74 | -- |

OPT-125M is the one case where ELSA comes out ahead: its unstructured paper
number beats EHWS. Against ELSA under the same row constraint, EHWS is lower.

### Zero-shot accuracy

At 70-90% sparsity, five of the seven tasks (ARC-C, OBQA, RTE, Winogrande,
BoolQ) sit near chance or the majority-class rate for every method. This also
holds in the ELSA paper's own tables (Tables 7-8). ARC-Easy and HellaSwag carry
most of the signal:
- **Gated DeltaNet-1.3B at 80%:** ELSA leads on ARC-Easy by 0.06, about 4
  standard errors. This is the one clear gap.
- **HGRN-1.3B at 70% and 90%:** EHWS leads on ARC-Easy.
- **Everywhere else:** the differences are within noise.

### Row balance: why semi-structured matters

ELSA's unstructured models are not balanced across rows. The table gives the
densest row's nonzeros divided by the mean row's:

| ELSA unstructured model | Mean over layers | Worst layer |
|---|---|---|
| HGRN-1.3B @ 30% | 1.27x | 1.42x |
| HGRN-1.3B @ 50% | 1.59x | 1.98x |
| HGRN-1.3B @ 70% | 2.26x | 3.28x |
| HGRN-1.3B @ 80% | 3.31x | 4.90x |
| HGRN-1.3B @ 90% | 5.18x | 9.73x |
| Gated DeltaNet-1.3B @ 80% | 2.47x | 4.41x |
| Gated DeltaNet-340M @ 80% | 2.58x | 4.37x |
| Mamba-1.3B @ 80% | 2.47x | 4.64x |

Every EHWS and ELSA row-wise model is exactly 1.00x. At 90%, HGRN's starved
rows concentrate in the recurrent gates: 3.0% of the forget-gate (`f_proj`)
rows fall below a quarter of the mean density, and 6 of them are empty.

### Pruned models

All pruned weights, both ours and the ELSA baselines, are on Hugging Face at
[huggingface.co/ikimyaii](https://huggingface.co/ikimyaii). Each repo has a
model card with the method, settings, results and a row-balance check. Repo
names give the model, size, sparsity pattern, method and sparsity. For
example:
- [HGRN-1.3B-semi-structured-EHWS-80pct-v1](https://huggingface.co/ikimyaii/HGRN-1.3B-semi-structured-EHWS-80pct-v1)
- [HGRN-1.3B-semi-structured-EHWS-90pct](https://huggingface.co/ikimyaii/HGRN-1.3B-semi-structured-EHWS-90pct)
- [HGRN-1.3B-unstructured-ELSA-90pct-tuned](https://huggingface.co/ikimyaii/HGRN-1.3B-unstructured-ELSA-90pct-tuned)

The `results.json` for the HGRN 80% run is in `results/_s16_hgrn/`.

### Still running

- Gated DeltaNet-1.3B / 340M and Mamba-1.3B at 50 / 70 / 90%, each with both
  ELSA baselines.
- Phase 1 to 70% on Mamba-1.3B and Gated DeltaNet-340M at 90%, and on every
  model at 80%.
- ELSA re-runs with lambda 1e-3 at 80%.
- 2.7B models are deferred: they do not fit on one 40 GB A100 with float32
  master weights.

## How the method works

Every prunable layer is an `nn.Linear` inside a model block. Embeddings and
the output head stay dense.

### Phase 1: prune each layer on its own (ALPS)

Phase 1 goes through the layers in the order the model runs them. For each
layer it asks: *which weights can I remove so that this layer's output
changes as little as possible?*

1. Run 256 calibration sequences through the model and record the inputs `X`
   reaching this layer. From them, build `H = (2/N) X Xᵀ`. `H` describes which
   input directions matter for this layer.
2. Find sparse weights `W` that keep the output close to the original
   weights `W0`. Formally, minimize `‖(W − W0) X‖²` with each row keeping
   exactly `k` weights.
3. Solve step 2 with ADMM, as ALPS does. ADMM alternates two easy
   sub-problems:
   - **x-step:** find the best weights while ignoring sparsity, plus a penalty
     `ρ` for being far from the current sparse guess. This has an exact
     closed-form solution, so no training or backprop is needed.
   - **z-step:** project onto the sparsity pattern. Each row keeps its `k`
     weights with the largest `H_ii · w_i²`.

   The penalty `ρ` grows from small to large (0.01 to 10 × mean of `diag(H)`,
   over 50 rounds). Early rounds explore which weights to keep. Late rounds
   force the dense and sparse solutions to agree.
4. Keep the chosen positions and re-solve for their values exactly (a
   least-squares refit), so the surviving weights compensate for the removed
   ones.
5. Write the pruned weights into the model, then move to the next layer. The
   next layer therefore sees inputs from an already-pruned network.

On HGRN-1.3B, Phase 1 takes about 10 minutes. It uses no teacher and no
gradients.

### Phase 2: train the whole pruned model (ELSA-style ADMM)

Phase 2 makes every pruned layer trainable at once and trains on real text.
It repeats these three steps:

```
x ← train for 32 steps on  f(x) + (λ/2)‖x − z + u‖²   (learn, while staying near the sparse z)
z ← keep the top-k weights per row of (x + u)          (project onto the sparsity pattern)
u ← u + x − z                                          (track the remaining disagreement)
```

- The loss is `f = 0.5 · cross-entropy + 0.5 · KD`. KD is a distillation term
  that matches the original dense model's predictions. No labels are needed.
- There are 128 rounds of 32 steps, so 4,096 Adam steps at batch size 8.
- The training data is ELSA's own 32,768 C4 sequences, each seen once.
- At the end, the weights are set to `z`. The model is exactly sparse, and
  every row has the same number of nonzeros.

Phase 2 also uses four options that matter on HGRN:
- `--master-fp32`: weights and Adam state are kept in float32, and the
  forward pass runs in bf16.
- `--p2-train-extra`: norms, embeddings and the output head are also trained,
  without a sparsity penalty.
- `--p2-hessian-refresh`: `diag(H)` is re-measured during training, so the
  z-step's scores follow the model as it changes.
- `--p2-epoch-sampling`: each training sequence is used once.

## Reproducing the HGRN-1.3B result

You need one GPU with 40 GB of memory (peak use is about 36 GB) and about
4 hours.

All commands start from the root of this repository.

**1. Install** (Python 3.11):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r semi-structured/requirements.txt
```

**2. Build the training data.** Phase 2 trains on exactly the tokens ELSA
uses. Build them with ELSA's own code (about 35 minutes, CPU only):

```bash
git clone https://github.com/log-postech/elsa.git
cd elsa        # use ELSA's environment (see its README)
python ../semi-structured/scripts/build_elsa_train_cache.py --model fla-hub/hgrn-1.3B-100B
cd ..
```

It prints the path to pass in the next step, e.g.
`elsa/dataset/fla-hub_hgrn-1.3B-100B/train/c4_32768_2048_hub`.

**3. Run:**

```bash
cd semi-structured
bash scripts/run_hgrn.sh ../elsa/dataset/fla-hub_hgrn-1.3B-100B/train/c4_32768_2048_hub
```

The script runs Phase 1, Phase 2 and the evaluation. It writes
`results/hgrn-1.3b-0.8/results.json` and the pruned weights to
`results/hgrn-1.3b-0.8/model-s0.8/`. Compare `wikitext2_ppl_elsa` and
`c4_ppl_elsa` against the table above.

For a different sparsity, pass it as the second argument, e.g.
`bash scripts/run_hgrn.sh <data dir> 0.9`. The other settings stay the same.

### The HGRN settings, and why each one is needed

`scripts/run_hgrn.sh` contains the full command. These are the settings that
matter:

| Setting | Value | Why |
|---|---|---|
| `--p1-mode local` | ALPS Phase 1 | Fast and accurate per-layer pruning. The older `global` mode takes hours and does worse. |
| `--hessian-n-calib` | 256 | Calibration sequences used to build each `H`. |
| `--no-auto-hparams` | | **Required.** Without it, the code uses a built-in table (`ehws/hparams.py`) with the wrong learning rate for this setup, and the result is 2–3× worse with no warning. |
| `--p2-lr` | 6e-4 | Phase 2 learning rate. This is 3× ELSA's value, to make up for the KD term. |
| `--p2-lambda-max`, `--p2-lambda-schedule` | 0.01, constant | ADMM penalty strength, the same as ELSA's HGRN run. |
| `--adam-beta2` | 0.95 | Same as ELSA. The PyTorch default (0.999) is worse here. |
| `--elsa-train-cache`, `--p2-epoch-sampling` | ELSA's data, each sequence once | Same training data as ELSA. |
| `--p2-train-extra` | on | Also trains the non-pruned parameters, as ELSA does. |
| `--p2-hessian-refresh` | on | Updates the z-step's `diag(H)` during training. |
| `--dtype bfloat16 --master-fp32` | | bf16 teacher and forward pass, float32 weights and optimizer. Without `--master-fp32`, small updates are lost to bf16 rounding. |
| `--gradient-checkpointing`, `--p2-micro-batch 1 --p2-grad-accum 8` | | Memory only, so it fits in 40 GB. The effective batch size is still 8. |

## OPT-125M

Same method, with two settings changed for this model: `lambda 3e-3`, and
`lr 4e-4` with Adam's default `beta2`. It runs in float32 and uses ELSA's
OPT-125M data, built with the same script and `--model facebook/opt-125m`.

```bash
python run_pipeline.py --model facebook/opt-125m \
  --sparsities 0.8 --seqlen 2048 --n-eval-c4 256 --dtype float32 \
  --p1-mode local --hessian-n-calib 256 \
  --elsa-train-cache <path>/facebook_opt-125m/train/c4_32768_2048_hub \
  --p2-epoch-sampling --p2-train-extra --p2-hessian-refresh \
  --no-auto-hparams --p2-lr 4e-4 --p2-lambda-max 3e-3 --p2-lambda-schedule constant --adam-beta2 0.999 \
  --skip-zeroshot --save-model --out results/opt-125m-0.8
```

## Evaluation

Each run reports perplexity two ways in `results.json`:
- `wikitext2_ppl_elsa` / `c4_ppl_elsa`: ELSA's exact protocol (its test sets,
  formula and float16 evaluation). **Use these to compare with ELSA.**
- `wikitext2_ppl` / `c4_ppl`: the full WikiText-2 test set and a 256-sequence
  slice of C4 validation.

Zero-shot accuracy covers ARC-Easy, ARC-Challenge, BoolQ, HellaSwag,
OpenBookQA, RTE and Winogrande, run with `lm-evaluation-harness`.

`eval_external.py` re-scores any saved model (ours or ELSA's). It also checks
the sparsity and that every row has the same number of nonzeros:

```bash
python eval_external.py --model fla-hub/hgrn-1.3B-100B \
  --weights results/hgrn-1.3b-0.8/model-s0.8/model.safetensors \
  --calib-cache results/hgrn-1.3b-0.8/cache --out results/rescore
```

## Code layout

```
run_pipeline.py          # end to end: dense eval -> Phase 1 -> Phase 2 -> eval -> results.json
eval_external.py         # re-score a saved pruned model
scripts/
  run_hgrn.sh            # the HGRN-1.3B command with the settings above
  build_elsa_train_cache.py  # builds ELSA's training data with ELSA's code
ehws/
  admm.py                # Phase 1 (local/ALPS and global) and Phase 2
  diagonal_projection.py # z-step: keep top-k per row by H_ii * w_i^2
  hessian.py             # H = (2/N) X Xᵀ from calibration inputs
  losses.py              # cross-entropy + KD loss
  model_layers.py        # finds the prunable layers of any decoder model, in run order
  calibration.py         # C4 / WikiText-2 data, including ELSA's versions
  perplexity.py          # both perplexity protocols
  zeroshot.py            # lm-evaluation-harness wrapper
  hparams.py             # default (lr, lambda) table; bypassed by --no-auto-hparams
  obs_projection.py      # alternative z-steps, for ablations only
results/_s16_hgrn/       # the exact launch config and results.json of the HGRN result above
```

The code works with any decoder-only model built from `nn.Linear` layers
inside repeated blocks, including OPT, LLaMA and HGRN. It finds the layers by
tracing one forward pass, so no per-model wiring is needed.

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

HGRN-1.3B (`fla-hub/hgrn-1.3B-100B`) at 80% sparsity. Perplexity uses ELSA's
own evaluation protocol, so the rows are directly comparable. Lower is better.

| | Sparsity pattern | WikiText-2 | C4 |
|---|---|---|---|
| Dense model | none | 11.81 | 16.09 |
| **EHWS (this code)** | **semi-structured** | **46.49** | **34.58** |
| ELSA, official code | semi-structured (same constraint) | 58.79 | 40.30 |
| ELSA, official code | unstructured | 54.08 | 36.92 |

Zero-shot accuracy (7-task average): dense 0.438, EHWS 0.377. The
`results.json` for this run is in `results/_s16_hgrn/`, and the pruned weights
are on Hugging Face:
[ikimyaii/HGRN-1.3B-semi-structured-EHWS-ALPS-phase1-80pct](https://huggingface.co/ikimyaii/HGRN-1.3B-semi-structured-EHWS-ALPS-phase1-80pct).

OPT-125M at 80%: EHWS 66.65 / 49.45, ELSA under the same constraint
75.86 / 53.45 (see [OPT-125M](#opt-125m) below).

All results are single runs. Repeating the same run can move WikiText-2 by up
to about 5%; C4 moves much less.

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

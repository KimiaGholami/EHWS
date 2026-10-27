# OPT-125M experiments

A copy of `../semi-structured/` (code as of Session 17) for OPT-125M-only experiments. Changes tried here stay out of the main code until they prove themselves. See `../semi-structured/PROGRESS.md` (Sessions 17-19) for results and reasoning.

**Best result so far (Session 19, `results/_s19_opt125m_batchA/`, arm `lr4e-4`):** 66.65 / 49.45 WikiText-2 / C4 PPL at 80% semi-structured sparsity (ELSA protocol). ELSA official under the same row-wise constraint scores 75.86 / 53.45, and the ELSA paper's unstructured number is 65.30 / 47.74. The weights are on Hugging Face: [ikimyaii/OPT-125M-semi-structured-EHWS-80pct](https://huggingface.co/ikimyaii/OPT-125M-semi-structured-EHWS-80pct). The exact command is in `../semi-structured/README.md` ("Running OPT-125M"); it uses none of the options below, so it gives the same run from either folder.

Options this copy adds over `semi-structured/` (all off by default, so the default behavior is identical):

- `--zstep-hessian-power G`: Phase 2 Z-step score `H_ii**G * v**2` (1 = Diag(H), 0.5 = tempered, 0 = magnitude). Batch A: G=0.5 is within noise.
- `--p2-hessian-ema B`: smooth the refreshed diag(H) with an EMA instead of replacing it. Batch A: B=0.9 helped a little (-1.6% C4).
- `--p2-train-extra-scope {all,norms}`: with `--p2-train-extra`, train all non-pruned parameters or LayerNorms only. Batch A: `norms` helped (-1.5% C4), probably overlapping with the lower-lr gain.

Run it the same way as the main pipeline, from this directory: `python run_pipeline.py --model facebook/opt-125m ...`. The exact commands are in each `results/_s*/launch.sh`.

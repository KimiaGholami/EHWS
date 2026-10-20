#!/bin/bash
# EHWS semi-structured on HGRN-1.3B: the exact settings behind the 80% result
# (WikiText2 46.49 / C4 34.58, ELSA protocol). Run from semi-structured/:
#
#   bash scripts/run_hgrn.sh <path to ELSA's train cache> [sparsity, default 0.8] [output dir]
#
# Needs one 40 GB GPU (A100-40GB peaks at ~36 GB) and takes about 4 hours.
# Build the train cache first with scripts/build_elsa_train_cache.py (see README).
set -euo pipefail

ELSA_DATA="${1:?usage: bash scripts/run_hgrn.sh <ELSA train cache dir> [sparsity] [out dir]}"
SPARSITY="${2:-0.8}"
OUT="${3:-results/hgrn-1.3b-$SPARSITY}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_pipeline.py \
  --model fla-hub/hgrn-1.3B-100B \
  --sparsities "$SPARSITY" --zeroshot-sparsities "$SPARSITY" \
  --seqlen 2048 --n-eval-c4 256 \
  --p1-mode local --hessian-n-calib 256 \
  --elsa-train-cache "$ELSA_DATA" \
  --no-auto-hparams --p2-lr 6e-4 --p2-lambda-max 0.01 --p2-lambda-schedule constant --adam-beta2 0.95 \
  --p2-epoch-sampling --p2-train-extra --p2-hessian-refresh \
  --dtype bfloat16 --master-fp32 --gradient-checkpointing \
  --micro-batch 2 --grad-accum 4 --p2-micro-batch 1 --p2-grad-accum 8 \
  --calib-cache "$OUT/cache" \
  --save-model \
  --out "$OUT"

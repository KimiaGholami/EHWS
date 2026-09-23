#!/bin/bash
# Over-relaxed ADMM sweep on facebook/opt-125m @ 80% sparsity, two-phase
# kept intact (only Phase 2's update rule changes). Baseline to beat:
# standard ADMM, two-phase, 128 rounds = WT2 85.98 / C4 52.68 (matches
# relax_alpha=1.0 below, which is a regression check -- see
# admm_relaxed.py's docstring, already verified bit-identical on a smoke test).
set -e
VENV=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv
cd /home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured
OUT=/anvil/scratch/x-kgholami/ehws-opt125m-relax-sweep

COMMON="--model facebook/opt-125m --sparsities 0.8 --zeroshot-sparsities 0.8 \
  --n-calib 2048 --seqlen 2048 --n-eval-c4 256 \
  --calib-cache /anvil/scratch/x-kgholami/ehws-calib-cache"

# GPU 0: alpha=1.0, 128 rounds -- regression check, should reproduce the
# standard-ADMM baseline (85.98/52.68) almost exactly.
CUDA_VISIBLE_DEVICES=0 setsid nohup "$VENV/bin/python3" relaxed_admm/run_pipeline.py $COMMON \
  --relax-alpha 1.0 --p2-rounds 128 \
  --out "$OUT/relax1.0-rounds128" \
  > "$OUT/relax1.0-rounds128.log" 2>&1 < /dev/null &
disown

# GPU 1: alpha=1.5, 128 rounds -- does over-relaxation improve quality at the same budget?
CUDA_VISIBLE_DEVICES=1 setsid nohup "$VENV/bin/python3" relaxed_admm/run_pipeline.py $COMMON \
  --relax-alpha 1.5 --p2-rounds 128 \
  --out "$OUT/relax1.5-rounds128" \
  > "$OUT/relax1.5-rounds128.log" 2>&1 < /dev/null &
disown

# GPU 2: alpha=1.8, 128 rounds -- more aggressive relaxation (Boyd et al.'s upper typical value).
CUDA_VISIBLE_DEVICES=2 setsid nohup "$VENV/bin/python3" relaxed_admm/run_pipeline.py $COMMON \
  --relax-alpha 1.8 --p2-rounds 128 \
  --out "$OUT/relax1.8-rounds128" \
  > "$OUT/relax1.8-rounds128.log" 2>&1 < /dev/null &
disown

# GPU 3: alpha=1.5, 64 rounds (half budget) -- does relaxation let Phase 2
# converge faster, matching/beating the 128-round baseline in half the time?
CUDA_VISIBLE_DEVICES=3 setsid nohup "$VENV/bin/python3" relaxed_admm/run_pipeline.py $COMMON \
  --relax-alpha 1.5 --p2-rounds 64 \
  --out "$OUT/relax1.5-rounds64" \
  > "$OUT/relax1.5-rounds64.log" 2>&1 < /dev/null &
disown

sleep 3
echo "launched:"
pgrep -af "$VENV/bin/python3.*relaxed_admm/run_pipeline.py"

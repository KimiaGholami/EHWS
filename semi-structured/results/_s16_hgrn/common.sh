# Shared settings for Session 16 (sourced by launch.sh / watchdog.sh).
REPO=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss
CODE=$REPO/EHWS/semi-structured
PY=$REPO/Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3
OUT=/anvil/scratch/x-kgholami/ehws-exp-out/s16-hgrn
CACHE=/anvil/scratch/x-kgholami/ehws-calib-cache
ELSA_DATA=$REPO/ELSA-official/dataset/fla-hub_hgrn-1.3B-100B/train/c4_32768_2048_hub
STATUS=$CODE/results/_s16_hgrn/status.txt
export HF_HOME=/anvil/scratch/x-kgholami/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Best HGRN @80% two-phase config (Session 14 local arm: Phase 2 lr 6e-4, lambda 0.01 constant,
# beta2 0.95, H 256), plus items 1/2/3/6: fp32 master weights, ELSA's 32,768 training sequences
# seen once, non-pruned params trained, diag(H) refreshed during Phase 2.
ARGS="--model fla-hub/hgrn-1.3B-100B --sparsities 0.8 --zeroshot-sparsities 0.8 --seqlen 2048 --n-eval-c4 256 \
  --micro-batch 2 --grad-accum 4 --p2-micro-batch 1 --p2-grad-accum 8 --dtype bfloat16 --master-fp32 \
  --gradient-checkpointing --no-auto-hparams --p2-lr 6e-4 --p2-lambda-max 0.01 --p2-lambda-schedule constant \
  --adam-beta2 0.95 --hessian-n-calib 256 --p1-mode local --elsa-train-cache $ELSA_DATA --p2-epoch-sampling \
  --p2-train-extra --p2-hessian-refresh --calib-cache $CACHE"
RUN=local-fp32-elsadata-extra-hrefresh

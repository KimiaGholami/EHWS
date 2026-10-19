#!/bin/bash
# Session 18: OPT-125M @80%, local two-phase, Phase 2 lambda sweep. Runs ON THE GPU NODE.
# Identical to Session 17's run (82.31 / 55.69 ELSA protocol at lambda 1e-3) except --p2-lambda-max.
# 1 GPU -> both arms share GPU 0 (~14-19 GB each); 2+ GPUs -> one arm per GPU.
REPO=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss
CODE=$REPO/EHWS/opt125m
PY=$REPO/Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3
OUT=/anvil/scratch/x-kgholami/ehws-exp-out/s18-opt125m-lambda
STATUS=$CODE/results/_s18_opt125m_lambda/status.txt
export HF_HOME=/anvil/scratch/x-kgholami/hf-cache PYTHONUNBUFFERED=1
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
ARGS="--model facebook/opt-125m --sparsities 0.8 --n-eval-c4 256 --seqlen 2048 --dtype float32 \
  --p1-mode local --hessian-n-calib 256 \
  --elsa-train-cache $REPO/ELSA-official/dataset/facebook_opt-125m/train/c4_32768_2048_hub \
  --p2-epoch-sampling --p2-train-extra --p2-hessian-refresh --calib-cache /anvil/scratch/x-kgholami/ehws-calib-cache \
  --no-auto-hparams --p2-lr 6e-4 --p2-lambda-schedule constant --adam-beta2 0.999 --skip-zeroshot --save-model"
NG=$(nvidia-smi -L | wc -l)
say "start on $(hostname), $NG GPU(s)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader -l 30 > $OUT.gpu.log 2>&1 &
SMI=$!
arm() {  # arm NAME LAMBDA GPU
  mkdir -p $OUT/$1
  (cd $CODE && CUDA_VISIBLE_DEVICES=$3 $PY -u run_pipeline.py $ARGS --p2-lambda-max $2 --out $OUT/$1 > $OUT/$1/run.log 2>&1)
  say "$1 (lambda $2) exit=$?: $(grep 'ELSA protocol' $OUT/$1/run.log | tail -1)"
}
arm lam3e-3 3e-3 0 &
A1=$!
sleep 60   # stagger the startup (model load, cache reads) on the node's single CPU
arm lam1e-2 1e-2 $([ $NG -ge 2 ] && echo 1 || echo 0) &
A2=$!
# watchdog: every 2 min, flag errors / memory / temperature / ECC
( sleep 120
  while pgrep -f "s18-opt125m-lambda/lam" > /dev/null; do
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | while IFS=', ' read IDX MEM TEMP ECC; do
      [ "$MEM" -gt 39800 ] && say "WARN GPU $IDX memory ${MEM}MiB"
      [ "$TEMP" -gt 85 ] && say "WARN GPU $IDX temperature ${TEMP}C"
      [ "$ECC" != "0" ] && [ "$ECC" != "[N/A]" ] && say "WARN GPU $IDX uncorrected ECC errors $ECC"
    done
    for j in lam3e-3 lam1e-2; do
      E=$(grep -a -m1 -E 'Traceback|OutOfMemory|CUDA error|[^a-zA-Z_](nan|NaN)([^a-zA-Z]|$)' $OUT/$j/run.log 2>/dev/null)
      [ -n "$E" ] && [ ! -f $OUT/$j/.reported ] && { say "ERROR $j: $E"; touch $OUT/$j/.reported; }
    done
    echo "$(date '+%F %T') $(for j in lam3e-3 lam1e-2; do echo -n "$j: $(grep -a 'phase2\] round' $OUT/$j/run.log 2>/dev/null | tail -1 | cut -c10-120) | "; done)" >> $OUT.watchdog.log
    sleep 120
  done ) &
wait $A1 $A2
kill $SMI 2>/dev/null
say "both lambda arms finished"

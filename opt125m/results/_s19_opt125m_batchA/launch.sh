#!/bin/bash
# Session 19: OPT-125M @80% batch A, run ON THE GPU NODE (1 GPU, two runs at a time).
# Base = Session 18 lambda 3e-3 run (73.85 / 52.52 ELSA protocol); each arm changes ONE thing.
# Two lanes run in parallel on GPU 0; a lane only starts an arm if >= 4h of walltime remain.
REPO=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss
CODE=$REPO/EHWS/opt125m
PY=$REPO/Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3
OUT=/anvil/scratch/x-kgholami/ehws-exp-out/s19-opt125m-batchA
STATUS=$CODE/results/_s19_opt125m_batchA/status.txt
export HF_HOME=/anvil/scratch/x-kgholami/hf-cache PYTHONUNBUFFERED=1
mkdir -p $OUT
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
BASE="--model facebook/opt-125m --sparsities 0.8 --n-eval-c4 256 --seqlen 2048 --dtype float32 \
  --p1-mode local --hessian-n-calib 256 \
  --elsa-train-cache $REPO/ELSA-official/dataset/facebook_opt-125m/train/c4_32768_2048_hub \
  --p2-epoch-sampling --p2-train-extra --calib-cache /anvil/scratch/x-kgholami/ehws-calib-cache \
  --no-auto-hparams --p2-lr 6e-4 --p2-lambda-max 3e-3 --p2-lambda-schedule constant --adam-beta2 0.999 \
  --skip-zeroshot --save-model"
REFRESH="--p2-hessian-refresh"
declare -A ARM=(
  [lam2e-3]="$REFRESH --p2-lambda-max 2e-3"
  [lam5e-3]="$REFRESH --p2-lambda-max 5e-3"
  [lamcos1e-2]="$REFRESH --p2-lambda-max 1e-2 --p2-lambda-schedule cosine"
  [hema0.9]="$REFRESH --p2-hessian-ema 0.9"
  [norefresh]=""
  [hpow0.5]="$REFRESH --zstep-hessian-power 0.5"
  [extras-norms]="$REFRESH --p2-train-extra-scope norms"
  [p1t0.7]="$REFRESH --p1-target-sparsity 0.7"
  [lr4e-4]="$REFRESH --p2-lr 4e-4"
  [lr9e-4]="$REFRESH --p2-lr 9e-4"
)
LANE1="lam2e-3 lamcos1e-2 norefresh extras-norms lr4e-4"
LANE2="lam5e-3 hema0.9 hpow0.5 p1t0.7 lr9e-4"
left_sec() {  # remaining walltime of this node's job, in seconds
  local L=$(squeue -h -u $USER -w $(hostname -s) -o %L | head -1) d=0
  [[ $L == *-* ]] && { d=${L%%-*}; L=${L#*-}; }
  IFS=: read -a p <<< "$L"
  while [ ${#p[@]} -lt 3 ]; do p=(0 "${p[@]}"); done
  echo $(( d*86400 + 10#${p[0]}*3600 + 10#${p[1]}*60 + 10#${p[2]} ))
}
lane() {
  for a in $1; do
    if [ $(left_sec) -lt 14400 ]; then say "SKIP $a: under 4h of walltime left"; continue; fi
    mkdir -p $OUT/$a
    say "start $a: ${ARM[$a]}"
    (cd $CODE && CUDA_VISIBLE_DEVICES=0 $PY -u run_pipeline.py $BASE ${ARM[$a]} --out $OUT/$a > $OUT/$a/run.log 2>&1)
    say "end $a exit=$?: $(grep 'ELSA protocol' $OUT/$a/run.log | tail -1)"
  done
}
say "batch A start on $(hostname): $(nvidia-smi -L | wc -l) GPU(s), $(( $(left_sec) / 3600 ))h walltime left"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader -l 60 > $OUT/gpu.log 2>&1 &
SMI=$!
lane "$LANE1" & L1=$!
sleep 90   # stagger startups on the node's single CPU
lane "$LANE2" & L2=$!
( while kill -0 $L1 2>/dev/null || kill -0 $L2 2>/dev/null; do   # watchdog
    sleep 120
    read MEM TEMP ECC <<< "$(nvidia-smi --query-gpu=memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i 0 | tr ',' ' ')"
    [ "${MEM:-0}" -gt 39800 ] && say "WARN gpu memory ${MEM}MiB"
    [ "${TEMP:-0}" -gt 85 ] && say "WARN gpu temperature ${TEMP}C"
    [ "${ECC:-0}" != "0" ] && [ "$ECC" != "[N/A]" ] && say "WARN uncorrected ECC errors $ECC"
    for d in $OUT/*/; do
      E=$(grep -a -m1 -E 'Traceback|OutOfMemory|CUDA error|[^a-zA-Z_](nan|NaN)([^a-zA-Z]|$)' $d/run.log 2>/dev/null)
      [ -n "$E" ] && [ ! -f $d/.reported ] && { say "ERROR $(basename $d): $E"; touch $d/.reported; }
    done
    echo "$(date '+%F %T') mem=${MEM} temp=${TEMP} $(for d in $OUT/*/; do echo -n "$(basename $d):$(grep -a 'phase2\] round' $d/run.log 2>/dev/null | tail -1 | grep -o 'round [0-9]*') "; done)" >> $OUT/watchdog.log
  done ) &
wait $L1 $L2
kill $SMI 2>/dev/null
say "batch A finished"

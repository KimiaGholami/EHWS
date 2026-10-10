#!/bin/bash
# Session 17: OPT-125M @80%, run ON THE GPU NODE. Two runs share GPU 0 in parallel (~16GB each):
#  - ehws: local two-phase + Session 16 items 2/3/6 (ELSA's 32,768 sequences seen once, non-pruned
#    params trained, diag(H) refresh). Everything else as the Session 12/13 local r128 run
#    (80.26 / 58.11 ELSA protocol): auto hparams lr 6e-4 / lambda 1e-3 constant, beta2 0.999, fp32, H 256.
#    Item 1 (fp32 master) is moot: OPT-125M always trained in full fp32.
#  - elsa-unstructured: ELSA official, same settings as the row-wise run (75.86 / 53.45) except
#    --admm_comparison_group=layer (ELSA's default projection).
REPO=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss
CODE=$REPO/EHWS/semi-structured
OUT=/anvil/scratch/x-kgholami/ehws-exp-out/s17-opt125m
STATUS=$CODE/results/_s17_opt125m/status.txt
export HF_HOME=/anvil/scratch/x-kgholami/hf-cache PYTHONUNBUFFERED=1
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
mkdir -p $OUT/ehws $OUT/elsa-unstructured
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader -i 0 -l 30 > $OUT/gpu.log 2>&1 &
SMI=$!
say "start on $(hostname): ehws + elsa-unstructured on GPU 0"
(cd $CODE && CUDA_VISIBLE_DEVICES=0 $REPO/Extreme_Layer_Global_Pruning_Unstructured/.venv/bin/python3 -u run_pipeline.py \
  --model facebook/opt-125m --sparsities 0.8 --n-eval-c4 256 --seqlen 2048 --dtype float32 \
  --p1-mode local --hessian-n-calib 256 --elsa-train-cache $REPO/ELSA-official/dataset/facebook_opt-125m/train/c4_32768_2048_hub \
  --p2-epoch-sampling --p2-train-extra --p2-hessian-refresh --calib-cache /anvil/scratch/x-kgholami/ehws-calib-cache \
  --skip-zeroshot --save-model --out $OUT/ehws > $OUT/ehws/run.log 2>&1; say "ehws exit=$?: $(grep 'ELSA protocol' $OUT/ehws/run.log | tail -1)") &
(cd $REPO/ELSA-official && CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u main.py --model=facebook/opt-125m --seqlen=2048 \
  --sparsity_ratio=0.8 --sparsity_type=unstructured --admm_comparison_group=layer --dataset=c4 \
  --admm_steps=4096 --admm_interval=32 --admm_batch_size=1 --admm_gradient_accumulation_steps=8 \
  --admm_lr=2e-4 --admm_lmda=1e-3 --admm_lmda_schedule_mode=constant --admm_precision=bf16 \
  --admm_save_inputs=True --admm_save_path=$OUT/elsa-unstructured/elsa-save --save_model=True --eval_zero_shot=False --seed=0 \
  > $OUT/elsa-unstructured/run.log 2>&1; say "elsa-unstructured exit=$?: $(grep -a "('wikitext2'" $OUT/elsa-unstructured/run.log | tail -1 | grep -o '\[.*')") &
# watchdog: every 2 min, flag errors / memory / temperature / ECC / walltime
( while pgrep -f "s17-opt125m" | grep -qv $$; do
    sleep 120
    read MEM TEMP ECC <<< "$(nvidia-smi --query-gpu=memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i 0 | tr ',' ' ')"
    echo "$(date '+%F %T') mem=${MEM}MiB temp=${TEMP}C ecc=$ECC left=$(squeue -h -u $USER -w $(hostname -s) -o %L | head -1)" >> $OUT/watchdog.log
    [ "${MEM:-0}" -gt 39800 ] && say "WARN gpu memory ${MEM}MiB"
    [ "${TEMP:-0}" -gt 85 ] && say "WARN gpu temperature ${TEMP}C"
    for j in ehws elsa-unstructured; do
      E=$(grep -a -m1 -E 'Traceback|OutOfMemory|CUDA error' $OUT/$j/run.log)
      [ -n "$E" ] && [ ! -f $OUT/$j/.reported ] && { say "ERROR $j: $E"; touch $OUT/$j/.reported; }
    done
  done ) &
wait %2 %3 2>/dev/null; wait
kill $SMI 2>/dev/null
say "both OPT-125M runs finished"

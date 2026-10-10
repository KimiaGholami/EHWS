#!/bin/bash
# Runs ON THE GPU NODE. Stage 1: GPU smoke test (real HGRN, full Phase 2 memory, 3 short rounds,
# eval + save) with GPU memory sampled every 5s. Stage 2: the full run, only if stage 1 passed.
source "$(dirname "$0")/common.sh"
GPU=${GPU:-0}
mkdir -p $OUT/smoke $OUT/$RUN
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
cd $CODE

say "stage 1: GPU smoke test on $(hostname) GPU $GPU"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader -i $GPU -l 5 > $OUT/smoke/gpu.log 2>&1 &
SMI=$!
CUDA_VISIBLE_DEVICES=$GPU $PY -u run_pipeline.py $ARGS --p1-local-rounds 5 --p2-rounds 3 --p2-x-steps 8 \
  --skip-zeroshot --save-model --checkpoint-dir $OUT/smoke/ckpt --out $OUT/smoke > $OUT/smoke/run.log 2>&1
RC=$?
kill $SMI
PEAK=$(awk -F', ' '{v=$2; sub(/ MiB/,"",v); v+=0; if (v>m) m=v} END {print m+0}' $OUT/smoke/gpu.log)
say "smoke exit=$RC peak_gpu_mem=${PEAK}MiB"
if [ $RC -ne 0 ] || [ ! -f $OUT/smoke/results.json ] || [ "$PEAK" -gt 39500 ]; then
  say "SMOKE FAILED -- full run NOT launched. See $OUT/smoke/run.log"
  exit 1
fi
rm -rf $OUT/smoke/model-s0.8 $OUT/smoke/ckpt   # 5GB of throwaway weights

say "stage 2: full run $RUN on GPU $GPU"
nohup bash $CODE/results/_s16_hgrn/watchdog.sh > $OUT/$RUN/watchdog.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU $PY -u run_pipeline.py $ARGS --save-model --out $OUT/$RUN > $OUT/$RUN/run.log 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f $OUT/$RUN/results.json ]; then
  say "FULL RUN DONE: $($PY -c "import json;e=json.load(open('$OUT/$RUN/results.json'))['sparsity']['0.8'];print('WT2 %.2f / C4 %.2f (ELSA protocol)'%(e['wikitext2_ppl_elsa'],e['c4_ppl_elsa']))")"
else
  say "FULL RUN FAILED exit=$RC -- see $OUT/$RUN/run.log"
fi

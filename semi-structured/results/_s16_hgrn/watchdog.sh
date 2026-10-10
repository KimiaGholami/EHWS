#!/bin/bash
# Runs ON THE GPU NODE next to the full run; every 2 min logs health and flags problems in status.txt.
source "$(dirname "$0")/common.sh"
GPU=${GPU:-0}
LOG=$OUT/$RUN/run.log
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
sleep 60
while true; do
  PID=$(pgrep -f "run_pipeline.py.*--out $OUT/$RUN" | head -1)
  SMI=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i $GPU)
  read MEM TEMP ECC <<< "$(echo $SMI | tr ',' ' ')"
  LEFT=$(squeue -h -u $USER -w $(hostname -s) -o %L | head -1)
  echo "$(date '+%F %T') pid=${PID:-none} gpu_mem=${MEM}MiB temp=${TEMP}C ecc=${ECC} walltime_left=$LEFT last: $(grep '\[phase' $LOG | tail -1 | cut -c1-140)"
  [ "${MEM:-0}" -gt 39800 ] 2>/dev/null && say "WARN gpu memory ${MEM}MiB near the 40GB limit"
  [ "${TEMP:-0}" -gt 85 ] 2>/dev/null && say "WARN gpu temperature ${TEMP}C"
  [ "${ECC:-0}" != "0" ] && [ "${ECC}" != "[N/A]" ] && say "WARN uncorrected ECC errors: $ECC"
  ERR=$(grep -m1 -E 'Traceback|OutOfMemory|CUDA error|[^a-zA-Z](nan|NaN|inf)([^a-zA-Z]|$)' $LOG)
  [ -n "$ERR" ] && { say "ERROR in run.log: $ERR"; break; }
  myquota 2>/dev/null | awk '$1=="home" && $5+0>95 {print}' | grep -q . && say "WARN /home quota above 95%"
  [ -z "$PID" ] && { say "watchdog: run process gone, stopping"; break; }
  sleep 120
done

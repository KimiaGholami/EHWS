#!/bin/bash
OUT=/anvil/scratch/x-kgholami/ehws-opt125m-relax-sweep
LOG=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/semi-structured/relaxed_admm/results/_opt125m_relax_sweep/watchdog.log
declare -A PIDS=( [0]=652518 [1]=652519 [2]=652520 [3]=652521 )
declare -A NAMES=( [0]=relax1.0-rounds128 [1]=relax1.5-rounds128 [2]=relax1.8-rounds128 [3]=relax1.5-rounds64 )

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"; }

DMESG_START_LINE=$(dmesg 2>/dev/null | wc -l)
log "watchdog started for pids: ${PIDS[*]}, dmesg baseline=line $DMESG_START_LINE"

while true; do
  problem=""

  for gpu in 0 1 2 3; do
    pid=${PIDS[$gpu]}
    name=${NAMES[$gpu]}

    read -r mem_used mem_total util ecc <<< "$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr ',' ' ')"
    if [ -z "$ecc" ]; then
      problem="GPU $gpu ($name): nvidia-smi query failed"
    elif [ "$ecc" != "0" ]; then
      problem="GPU $gpu ($name): uncorrected ECC errors = $ecc"
    elif [ -n "$mem_used" ] && [ "$mem_used" -gt 35000 ] 2>/dev/null; then
      problem="GPU $gpu ($name): memory ${mem_used}/${mem_total}MiB -- close to the 40960MiB limit, OOM risk"
    else
      log "GPU $gpu ($name): mem=${mem_used}/${mem_total}MiB util=${util}%"
    fi
    [ -n "$problem" ] && break

    if kill -0 "$pid" 2>/dev/null; then
      if grep -qiE "CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)" "$OUT/$name.log" 2>/dev/null; then
        problem="GPU $gpu ($name): OOM/traceback signature found in run.log while pid $pid still alive"
      fi
    else
      if [ -f "$OUT/$name/results.json" ]; then
        log "GPU $gpu ($name): pid $pid exited, results.json present -- clean finish"
      else
        problem="GPU $gpu ($name): pid $pid exited with NO results.json -- died mid-run"
      fi
    fi
    [ -n "$problem" ] && break
  done

  if [ -n "$problem" ]; then
    log "PROBLEM DETECTED: $problem"
    echo "=== WATCHDOG: PROBLEM DETECTED ==="
    echo "$problem"
    echo "-- last 30 lines of ${NAMES[$gpu]}.log --"
    tail -n 30 "$OUT/${NAMES[$gpu]}.log" 2>/dev/null
    exit 1
  fi

  new_dmesg=$(dmesg 2>/dev/null | tail -n +"$((DMESG_START_LINE + 1))" | grep -iE "xid|out of memory|oom.kill" || true)
  if [ -n "$new_dmesg" ]; then
    log "PROBLEM DETECTED: new dmesg Xid/OOM lines:"
    echo "$new_dmesg" | while IFS= read -r line; do log "  dmesg: $line"; done
    echo "=== WATCHDOG: PROBLEM DETECTED (dmesg) ==="
    echo "$new_dmesg"
    exit 1
  fi

  all_dead=1
  for gpu in 0 1 2 3; do
    kill -0 "${PIDS[$gpu]}" 2>/dev/null && all_dead=0
  done
  if [ "$all_dead" -eq 1 ]; then
    log "all 4 processes finished, no compute/memory problems detected at any point"
    echo "=== WATCHDOG: sweep finished, no GPU compute/memory issues detected ==="
    exit 0
  fi

  sleep 60
done

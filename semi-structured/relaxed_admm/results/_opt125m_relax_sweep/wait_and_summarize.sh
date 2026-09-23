#!/bin/bash
OUT=/anvil/scratch/x-kgholami/ehws-opt125m-relax-sweep
declare -A PIDS=( [relax1.0-rounds128]=652518 [relax1.5-rounds128]=652519 [relax1.8-rounds128]=652520 [relax1.5-rounds64]=652521 )

while true; do
  all_done=1
  for name in "${!PIDS[@]}"; do
    pid=${PIDS[$name]}
    [ -f "$OUT/$name/results.json" ] && continue
    kill -0 "$pid" 2>/dev/null && all_done=0
  done
  [ "$all_done" -eq 1 ] && break
  sleep 60
done

echo "=== OPT-125M over-relaxed ADMM sweep: all arms finished ==="
for name in relax1.0-rounds128 relax1.5-rounds128 relax1.8-rounds128 relax1.5-rounds64; do
  echo "-- $name --"
  if [ -f "$OUT/$name/results.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT/$name/results.json'))
s = d['sparsity']['0.8']
print(f\"  WT2={s['wikitext2_ppl']:.2f} C4={s['c4_ppl']:.2f} achieved_sparsity={s['achieved_sparsity']:.4f}\")
if 'zeroshot' in s:
    print(f\"  zeroshot_avg={s['zeroshot']['average']:.4f}\")
"
  else
    echo "  FAILED / no results.json -- last 20 lines of log:"
    tail -n 20 "$OUT/$name.log"
  fi
done

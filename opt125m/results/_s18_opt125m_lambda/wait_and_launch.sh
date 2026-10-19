#!/bin/bash
# Runs ON THE LOGIN NODE (detached). Waits for the OnDemand job to start, then starts launch.sh on its node.
JOB=${1:-21236984}
CODE=/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS/opt125m
STATUS=$CODE/results/_s18_opt125m_lambda/status.txt
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
say "waiting for job $JOB to start"
while true; do
  ST=$(squeue -h -j $JOB -o %T 2>/dev/null)
  [ "$ST" = "RUNNING" ] && break
  if [ -z "$ST" ]; then
    MISS=$((${MISS:-0} + 1))
    [ $MISS -ge 3 ] && { say "job $JOB is no longer in the queue -- giving up"; exit 1; }
  else
    MISS=0
  fi
  sleep 60
done
NODE=$(squeue -h -j $JOB -o %N)
say "job $JOB RUNNING on $NODE; starting launch.sh there in 2 min"
sleep 120
ssh -o BatchMode=yes -o StrictHostKeyChecking=no $NODE \
  "setsid nohup bash $CODE/results/_s18_opt125m_lambda/launch.sh > $CODE/results/_s18_opt125m_lambda/launch.log 2>&1 < /dev/null &" \
  && say "launch.sh started on $NODE" || say "FAILED to ssh to $NODE"

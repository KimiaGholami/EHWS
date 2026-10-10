#!/bin/bash
# Runs ON THE LOGIN NODE (detached). Waits for the OnDemand job to start, then starts launch.sh on its node.
JOB=${1:-21177562}
source "$(dirname "$0")/common.sh"
say() { echo "$(date '+%F %T') $*" | tee -a $STATUS; }
say "waiting for job $JOB to start"
while true; do
  ST=$(squeue -h -j $JOB -o %T 2>/dev/null)
  [ "$ST" = "RUNNING" ] && break
  [ -z "$ST" ] && { say "job $JOB is no longer in the queue (state empty) -- giving up"; exit 1; }
  sleep 60
done
NODE=$(squeue -h -j $JOB -o %N)
say "job $JOB RUNNING on $NODE; starting launch.sh there in 2 min"
sleep 120
ssh -o BatchMode=yes -o StrictHostKeyChecking=no $NODE \
  "setsid nohup bash $CODE/results/_s16_hgrn/launch.sh > $CODE/results/_s16_hgrn/launch.log 2>&1 < /dev/null &" \
  && say "launch.sh started on $NODE" || say "FAILED to ssh to $NODE"

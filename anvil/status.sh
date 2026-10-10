#!/bin/bash
# Show your queued/running Anvil jobs, or tail one job's log.
#   anvil/status.sh            list jobs with estimated start times
#   anvil/status.sh <jobid>    follow that job's log (Ctrl-C to stop)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

if [[ $# -eq 0 ]]; then
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "squeue -u \$USER -o '%.10i %.25j %.9T %.10M %.10l %.20S %R'"
else
  ssh "${SSH_OPTS[@]}" -t "$SSH_HOST" "tail -n 50 -f $REMOTE_DIR/anvil/logs/*-$1.out"
fi

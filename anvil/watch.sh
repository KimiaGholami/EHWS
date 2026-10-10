#!/bin/bash
# Poll an Anvil job and show a Mac notification when it starts and when it ends.
#   anvil/watch.sh <jobid> [poll seconds, default 120]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

JOBID="$1"; POLL="${2:-120}"

notify() {
  osascript -e "display notification \"$2\" with title \"Anvil job $JOBID\" sound name \"Glass\""
}

last=""
while true; do
  state=$(ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$SSH_HOST" \
    "sacct -j $JOBID -X -n -o State%20 | head -1" 2>/dev/null | awk '{print $1}')
  if [[ -n "$state" && "$state" != "$last" ]]; then
    case "$state" in
      RUNNING) notify "$state" "Running now. Log: anvil/logs/ on Anvil" ;;
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
        notify "$state" "Finished: $state"; exit 0 ;;
    esac
    last="$state"
  fi
  sleep "$POLL"
done

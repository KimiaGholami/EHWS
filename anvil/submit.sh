#!/bin/bash
# Queue a GPU run on Anvil from this Mac. It starts by itself as soon as a
# GPU frees up; you don't need to stay connected.
#
#   anvil/submit.sh [options] <semi-structured|unstructured> <run_pipeline.py args...>
#
# Options:
#   -t HH:MM:SS  walltime (default from config.sh; shorter usually starts sooner)
#   -g N         GPUs (default 1)
#   -n NAME      job name (default: value of --out, else the project name)
#   -s           first copy local code to Anvil (only files newer here; results/ etc. skipped)
#   -w           then watch the job and pop up Mac notifications when it starts/ends
#
# Example:
#   anvil/submit.sh -t 08:00:00 -w semi-structured \
#     --model facebook/opt-125m --sparsities 0.5 --out results/opt-125m
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

TIME="$DEFAULT_TIME"; GPUS="$DEFAULT_GPUS"; NAME=""; SYNC=0; WATCH=0
while getopts "t:g:n:sw" opt; do
  case $opt in
    t) TIME="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    s) SYNC=1 ;;
    w) WATCH=1 ;;
    *) sed -n '2,19p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then sed -n '2,19p' "$0"; exit 1; fi
PROJECT="$1"; shift
if [[ ! -d "$HERE/../$PROJECT" ]]; then echo "no such project folder: $PROJECT"; exit 1; fi
if [[ "$ACCOUNT" == *CHANGEME* ]]; then echo "set ACCOUNT in anvil/config.sh first"; exit 1; fi

if [[ -z "$NAME" ]]; then
  NAME="$PROJECT"
  for ((i = 1; i <= $#; i++)); do
    if [[ "${!i}" == "--out" ]]; then j=$((i + 1)); NAME="$(basename "${!j}")"; fi
  done
fi

if [[ $SYNC -eq 1 ]]; then
  # -u: never overwrite a file that's newer on Anvil.
  rsync -avu -e "ssh ${SSH_OPTS[*]}" \
    --exclude results/ --exclude elsa-out/ --exclude ELSA-official/ \
    --exclude __pycache__/ --exclude '*.docx' --exclude '*.pdf' --exclude .DS_Store \
    "$HERE/../$PROJECT/" "$SSH_HOST:$REMOTE_DIR/$PROJECT/"
fi

# Always ship these helper scripts so job.sub on Anvil is current.
rsync -aq -e "ssh ${SSH_OPTS[*]}" --exclude logs/ "$HERE/" "$SSH_HOST:$REMOTE_DIR/anvil/"

# Quote every pipeline arg so it survives the remote shell intact.
ARGS=$(printf '%q ' "$@")

JOBID=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "
  set -e
  mkdir -p $REMOTE_DIR/anvil/logs
  EHWS_REMOTE_DIR=$REMOTE_DIR EHWS_VENV=$VENV sbatch --parsable \
    -A $ACCOUNT -t $TIME --gpus-per-node=$GPUS -J $NAME \
    -o $REMOTE_DIR/anvil/logs/%x-%j.out \
    $REMOTE_DIR/anvil/job.sub $PROJECT $ARGS
")

echo "submitted job $JOBID ($NAME)"
echo "log on Anvil: $REMOTE_DIR/anvil/logs/$NAME-$JOBID.out"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "squeue -j $JOBID --start -o '%.10i %.20j %.8T %.20S  (estimated start)'" || true

if [[ $WATCH -eq 1 ]]; then
  nohup "$HERE/watch.sh" "$JOBID" >/dev/null 2>&1 &
  echo "watching in the background; you'll get a Mac notification when it starts and ends"
fi

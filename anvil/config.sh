# Settings shared by submit.sh / watch.sh. Edit ACCOUNT before first use.

# GPU allocation to charge. On Anvil run `mybalance` and copy the GPU one
# (usually ends in -gpu).
ACCOUNT="cis261249-gpu"

# SSH alias from ~/.ssh/config.
SSH_HOST="anvil-login"

# Where this EHWS folder lives on Anvil.
REMOTE_DIR="/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/EHWS"

# Python env with the pinned requirements (see semi-structured/PROGRESS.md,
# "Environment": Anvil's pip can't install them fresh, so reuse this one).
VENV="/home/x-kgholami/Weight-Sparsity-Using-Real-Loss/Extreme_Layer_Global_Pruning_Unstructured/.venv"

# Defaults, overridable per job with -t / -g.
DEFAULT_TIME="16:00:00"
DEFAULT_GPUS=1

# Reuse one SSH connection for all calls in a script run.
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=10m)

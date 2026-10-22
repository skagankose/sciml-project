# Source from the repository root on HPC3 (interactive node or inside sbatch):
#   source cluster/env.sh
# Loads conda, activates the project env and keeps every cache off the 50 GB $HOME.
# Do not `set -u` before sourcing: conda's activate scripts use unset variables.

CONDA_MODULE=${CONDA_MODULE:-miniconda3/24.9.2}
CONDA_INIT=${CONDA_INIT:-$HOME/.mycondainit-24.9.2}
ENV_NAME=${ENV_NAME:-well-rb}
PUB=/pub/$USER

module load "$CONDA_MODULE"
# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate "$ENV_NAME"

export PYTHONNOUSERSITE=1
export DATA_ROOT=${DATA_ROOT:-$PUB/data/the_well}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$PUB/projects/well-rb-four-families/outputs}
export XDG_CACHE_HOME=$PUB/.cache
export PIP_CACHE_DIR=$PUB/.cache/pip
export MPLCONFIGDIR=$PUB/.cache/matplotlib
export HF_HOME=$PUB/.cache/huggingface
export TORCH_HOME=$PUB/.cache/torch
export WANDB_MODE=disabled            # neuraloperator imports wandb; nothing is logged
export WANDB_DIR=$PUB/.cache/wandb
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

#!/bin/bash
# Create the `well-rb` conda env on HPC3 and record exactly what was installed.
# Run on a CPU compute node from the repo root (RCIC: installs fail on login nodes), as a
# batch job (interactive sessions were too slow to use):
#   cd /pub/$USER/projects/well-rb-four-families && mkdir -p logs && sbatch cluster/slurm/setup_cpu.sbatch
# Assumes the one-time conda setup from the HPC3 guide (envs/pkgs dirs on /pub,
# ~/.mycondainit-24.9.2) is done.
set -eo pipefail

CONDA_MODULE=${CONDA_MODULE:-miniconda3/24.9.2}
CONDA_INIT=${CONDA_INIT:-$HOME/.mycondainit-24.9.2}
ENV_NAME=${ENV_NAME:-well-rb}
export PIP_CACHE_DIR=/pub/$USER/.cache/pip
# conda's package cache on node-local disk. With the cache on /pub, conda's own downloads came
# back empty (ChecksumMismatchError, actual sha256 e3b0c442..., the empty-file hash) and every
# failed run left *.partial files that broke the next one (2026-11-07). The env itself still
# lands in /pub/$USER/conda/envs.
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${TMPDIR:-/tmp}/conda-pkgs-${SLURM_JOB_ID:-$$}}
mkdir -p "$CONDA_PKGS_DIRS"

module load "$CONDA_MODULE"
# shellcheck disable=SC1090
source "$CONDA_INIT"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME exists; updating packages only"
else
    conda env create -f environment.yml -n "$ENV_NAME"
fi
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

pip install -r requirements-cuda.txt          # torch 2.8.0 + cu128 (sm_80 for A100)
pip install -r requirements-common.txt        # shared exact pins (incl. pinned The Well commit)
pip install -e . --no-deps
conda clean -a -f -y

mkdir -p env
conda env export > env/hpc3-conda.yml
pip freeze > env/hpc3-pip.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list())"
WANDB_MODE=disabled python -m rbbench models --size full
echo "done; commit env/hpc3-*.{yml,txt} from the Mac after pulling them back (cluster/pull.sh --env)"

#!/bin/bash
# Create the `well-rb` conda env on HPC3 and record exactly what was installed.
# Run on a CPU compute node from the repo root (RCIC: installs fail on login nodes):
#   srun -c 4 -p free --mem=16G --time=02:00:00 --pty /bin/bash -i
#   cd /pub/$USER/projects/well-rb-four-families && bash cluster/setup_env.sh
# Assumes the one-time conda setup from the HPC3 guide (envs/pkgs dirs on /pub,
# ~/.mycondainit-24.9.2) is done.
set -eo pipefail

CONDA_MODULE=${CONDA_MODULE:-miniconda3/24.9.2}
CONDA_INIT=${CONDA_INIT:-$HOME/.mycondainit-24.9.2}
ENV_NAME=${ENV_NAME:-well-rb}
export PIP_CACHE_DIR=/pub/$USER/.cache/pip

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

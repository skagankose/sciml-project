#!/bin/bash
# Create the `rb-dedalus` conda env on HPC3 (numerical-solver baseline) and check the solver.
# Run as a batch job from the repo root (installs fail on login nodes):
#   mkdir -p logs && sbatch cluster/slurm/setup_dedalus_cpu.sbatch
# Same conventions as cluster/setup_env.sh: the env lands in /pub/$USER/conda/envs, the package cache on
# node-local disk (conda's downloads into /pub came back empty on 2026-11-07).
set -eo pipefail

CONDA_MODULE=${CONDA_MODULE:-miniconda3/24.9.2}
CONDA_INIT=${CONDA_INIT:-$HOME/.mycondainit-24.9.2}
ENV_NAME=${ENV_NAME:-rb-dedalus}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-${TMPDIR:-/tmp}/conda-pkgs-${SLURM_JOB_ID:-$$}}
mkdir -p "$CONDA_PKGS_DIRS"

module load "$CONDA_MODULE"
# shellcheck disable=SC1090
source "$CONDA_INIT"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME exists; updating it"
    conda env update -f environment-dedalus.yml -n "$ENV_NAME"
else
    conda env create -f environment-dedalus.yml -n "$ENV_NAME"
fi
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
conda clean -a -f -y

mkdir -p env
conda env export > env/hpc3-dedalus-conda.yml
python -c "import dedalus, numpy, scipy, h5py; from importlib.metadata import version; print('dedalus', version('dedalus'))"
# grids, restart reproducibility, float32 storage and a resolution change; exits non-zero on failure
python scripts/solver_rollouts.py --selftest
echo "done; pull env/hpc3-dedalus-conda.yml (cluster/pull.sh --env) and commit it from the Mac"

#!/bin/bash
# One-off environment check on a real A100: GPU/torch info, full test suite (including the
# CUDA tests at 512x128 for all four models) and the synthetic smoke pipeline on CUDA.
#   mkdir -p logs && sbatch cluster/slurm/check.sbatch
#SBATCH --job-name=rb4-check
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

echo "host $(hostname)  job $SLURM_JOB_ID  commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m rbbench check-gpu --require A100
python -m pytest -q
python -m pytest -q -m cuda -rs
python -m rbbench smoke --device cuda --synthetic --grid 64 32 --steps 3 --all-models \
    --workdir "/pub/$USER/tmp/rb4-smoke-$SLURM_JOB_ID"
echo "CHECK PASSED"

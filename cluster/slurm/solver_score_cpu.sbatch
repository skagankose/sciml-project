#!/bin/bash
# Score the solver rollouts (outputs/solver/<res>/) with the models' metrics, in the well-rb env (CPU code).
# Class account, paid `gpu` partition only (owner, 2026-11-21); the GPU it holds is idle (32 + 4 SU/h).
# Writes outputs/baselines/solver_<res>/evaluations/. Rerun after more rollouts finish; it rescans everything.
#   sbatch cluster/slurm/solver_score_cpu.sbatch
#SBATCH --job-name=rb4-solver-score
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID}  commit $(git rev-parse HEAD)"
python -m rbbench eval-solver

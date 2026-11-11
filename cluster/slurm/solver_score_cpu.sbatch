#!/bin/bash
# Score the solver rollouts (outputs/solver/<res>/) with the models' metrics, in the well-rb env, on a CPU node.
# Writes outputs/baselines/solver_<res>/evaluations/. Rerun after more rollouts finish; it rescans everything.
#   sbatch cluster/slurm/solver_score_cpu.sbatch
#SBATCH --job-name=rb4-solver-score
#SBATCH --partition=free
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

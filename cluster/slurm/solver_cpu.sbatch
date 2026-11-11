#!/bin/bash
# Numerical-solver baseline: Dedalus rollouts from the cohort's start frames, one (Ra, Pr) regime per array task
# (35 regimes, the first cohort trajectory of each, starts 20 / 80 / 140), one CPU core each, free partition.
# Finished rollouts are kept, so a preempted task resumes when resubmitted. One resolution per submission:
#   sbatch --array=0-34 cluster/slurm/solver_cpu.sbatch                              # 512x128 (the data's)
#   sbatch --array=0-34 --export=ALL,RES=256x64 cluster/slurm/solver_cpu.sbatch
#   sbatch --array=0-34 --export=ALL,RES=128x32 cluster/slurm/solver_cpu.sbatch
# Then score them in well-rb: sbatch cluster/slurm/solver_score_cpu.sbatch
#SBATCH --job-name=rb4-solver
#SBATCH --partition=free
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV_NAME=rb-dedalus source cluster/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)  res ${RES:-512x128}"
python scripts/solver_rollouts.py --resolution "${RES:-512x128}"

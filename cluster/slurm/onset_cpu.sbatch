#!/bin/bash
# Onset frame of every validation and test trajectory from its true kinetic energy (D28, D29; docs/final_plan.md WP4).
# Reads velocity only (~18 GB per split) and writes <OUTPUT_ROOT>/onset/{onset,ke}_{valid,test}.{json,csv}.
# No model output is read. Pull with cluster/pull.sh, check the definition on validation, then commit the tables.
# Class account, paid `gpu` partition only (owner, 2026-11-21); CPU work, so the GPU it holds is idle (32 + 2 SU/h).
#   sbatch cluster/slurm/onset_cpu.sbatch
#SBATCH --job-name=rb4-onset
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID}  commit $(git rev-parse HEAD)"
python -m rbbench onset-frames --splits valid test

#!/bin/bash
# Our trained runs vs the paper (decisions D20): one-step VRMSE of a run's milestone checkpoint on every window
# of valid and test, under our protocol and the paper's. Run eval-released first: its released-checkpoint
# scores are the reference. One manifest row per array index, as in train_a100.sbatch:
#   sbatch --array=<i>,<j> --export=ALL,MANIFEST=tasks/train.jsonl,LABEL=4h cluster/slurm/paper_compare_a100.sbatch
# The job exits 1 (FAILED in sacct) when the run is worse than the paper; the log says which check.
#SBATCH --job-name=rb4-paper
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)"
python -m rbbench check-gpu --require A100 > /dev/null
exec python -m rbbench compare-paper --manifest "${MANIFEST:-tasks/train.jsonl}" \
    --task-index "$SLURM_ARRAY_TASK_ID" --label "${LABEL:-4h}" --device cuda

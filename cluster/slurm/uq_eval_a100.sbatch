#!/bin/bash
# Uncertainty quantification and noise robustness (D30, D17; docs/final_plan.md WP8, WP9): one ensemble of
# tasks/uq_ensembles.json per array task, through `rbbench eval-uq`, on the validation split (calibration, clean) and
# the test split (clean; the 4 h seed pairs and persistence also at noise 0.01-0.20). Writes
# <OUTPUT_ROOT>/uq/<ensemble>/<split>/; finished splits are skipped, so a resubmission resumes.
#   0-5  the two seeds of each core configuration at 4 h (CNextU-Net, TFNO, AViT x 10%, 100%)
#   6    persistence
#   7-8  the beta-NLL CNextU-Net ensembles (tasks/uq.jsonl), 10% and 100%, 1 h
#   9-10 their MSE baselines with the same 1 h schedule (tasks/train_1h.jsonl, tasks/train_1h_f100.jsonl)
# Paid partition (owner approval, 2026-11-21):
#   sbatch -p gpu -A eecs195221_class_gpu --array=0,2,4,6 cluster/slurm/uq_eval_a100.sbatch   # 10% pairs, persistence
#   sbatch -p gpu -A eecs195221_class_gpu --array=1,3,5 cluster/slurm/uq_eval_a100.sbatch     # after the 100% runs
#   sbatch -p gpu -A eecs195221_class_gpu --array=7-10 cluster/slurm/uq_eval_a100.sbatch      # after the UQ runs
# SPLITS="valid" or "test" runs one split only.
#SBATCH --job-name=rb4-uq
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export PYTHONUNBUFFERED=1

echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)"
python -m rbbench check-gpu --require A100 > /dev/null
EXTRA=()
if [ -n "${SPLITS:-}" ]; then
    read -r -a SPLIT_LIST <<< "$SPLITS"
    EXTRA=(--splits "${SPLIT_LIST[@]}")
fi
exec python -m rbbench eval-uq --index "$SLURM_ARRAY_TASK_ID" --device cuda ${EXTRA[@]+"${EXTRA[@]}"}

#!/bin/bash
# Main sweep: one 4 h (training-loop time) run per task row, one A100 each, one at a time.
# DO NOT SUBMIT without owner approval (Phase I gate). Submit from the repo root:
#   mkdir -p logs && sbatch cluster/slurm/train_a100.sbatch
# Paid partition only with permission:  sbatch -p gpu -A <LAB>_gpu cluster/slurm/train_a100.sbatch
# Resubmitting the same array is safe: complete tasks exit at once, partial ones resume.
# --time is a job ceiling, not the 4 h training target. The budget counts GPU training time only (D24), so a
# run that waits for data from /pub takes longer in real time (AViT ~7 h for its 4 h); validation, checkpoints
# and startup add more. If the ceiling is reached the run saves its state: resubmit the same task.
#SBATCH --job-name=rb4-train
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=10:00:00
#SBATCH --array=0-11%1
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
# DataLoader workers: all CPUs of the job but one (runtime settings never change a run's identity).
export RBBENCH_NUM_WORKERS=${RBBENCH_NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-5} - 1 ))}
MANIFEST=${MANIFEST:-tasks/train.jsonl}

echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m rbbench check-gpu --require A100 > /dev/null
# exec: python replaces this shell, so Slurm's USR1 (5 min before the limit) and SIGTERM
# (preemption) reach the training loop, which writes last_resume.pt and exits with code 3.
exec python -m rbbench train-task \
    --manifest "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" \
    --device cuda --resume-if-present --skip-if-complete ${EXTRA_ARGS:-}

#!/bin/bash
# Main sweep: one 4 h (training-loop time) run per task row, one A100 each.
# DO NOT SUBMIT without owner approval (Phase I gate). Submit from the repo root:
#   mkdir -p logs && sbatch cluster/slurm/train_a100.sbatch
# Paid partition only with permission:  sbatch -p gpu -A <LAB>_gpu cluster/slurm/train_a100.sbatch
# Resubmitting the same array is safe: complete tasks exit at once, partial ones resume.
# --time is a job ceiling, not the 4 h training target. The budget counts GPU training time only (D24), so a
# run that waits for data from /pub takes longer in real time (AViT ~7 h for its 4 h); validation, checkpoints
# and startup add more. If the ceiling is reached the run saves its state: resubmit the same task.
#
# STAGE_DATA=1 copies the training split to the node's local disk ($TMPDIR, 1.83 TB on the A100 nodes) before
# training and reads it from there (docs/final_plan.md WP1); the copy is deleted when the job ends. Samples and run
# ids are unchanged. If the copy fails, training reads from /pub as before. The 100% rows (owner, 2026-11-21):
#   sbatch -p gpu -A eecs195221_class_gpu --array=2,3,6,7,10,11 --time=20:00:00 --tmp=400G \
#       --export=ALL,STAGE_DATA=1 cluster/slurm/train_a100.sbatch
# A crash (e.g. a /pub error) restarts the run inside the allocation: cluster/run_with_restarts.sh.
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
if [ -n "${STAGE_DATA:-}" ]; then
    STAGE_DIR="${TMPDIR:-/tmp}/rbbench-data-${SLURM_JOB_ID}"
    trap 'rm -rf "$STAGE_DIR"' EXIT
    echo "staging the training split to $STAGE_DIR at $(date -u +%FT%TZ)"
    df -h "$(dirname "$STAGE_DIR")" | tail -1
    if python -m rbbench stage-data --dest "$STAGE_DIR" --splits train --workers "${STAGE_WORKERS:-4}"; then
        export DATA_ROOT="$STAGE_DIR" CLEANUP_DIR="$STAGE_DIR"
        echo "staged at $(date -u +%FT%TZ); training reads $DATA_ROOT"
    else
        rm -rf "$STAGE_DIR"
        echo "staging failed at $(date -u +%FT%TZ); training reads $DATA_ROOT"
    fi
    trap - EXIT
fi
# exec: the wrapper replaces this shell, so Slurm's USR1 (5 min before the limit) and SIGTERM (preemption) reach
# it; it forwards them to the training loop, which writes last_resume.pt and exits with code 3. A crash restarts
# the run inside this allocation (at most twice), and the wrapper deletes the staged copy when it exits.
exec bash cluster/run_with_restarts.sh python -m rbbench train-task \
    --manifest "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" \
    --device cuda --resume-if-present --skip-if-complete ${EXTRA_ARGS:-}

#!/bin/bash
# Evaluate the milestone checkpoints (1h and 4h) of each finished training task, separate from
# the training clock. Needs tasks/eval_cohort.json and outputs/audit/dataset_audit.json.
#   sbatch --array=0-11%2 cluster/slurm/eval_a100.sbatch
# Persistence baseline once:  sbatch --array=0 --export=ALL,BASELINE=1 cluster/slurm/eval_a100.sbatch
# A released checkpoint on the same cohort (rollouts, failure horizons; outputs/baselines/released/<model>/evaluations):
#   sbatch --array=0 --export=ALL,RELEASED=cnext_unet cluster/slurm/eval_a100.sbatch
#SBATCH --job-name=rb4-eval
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
MANIFEST=${MANIFEST:-tasks/train.jsonl}
AUDIT=${AUDIT:-outputs/audit/dataset_audit.json}

echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)"
python -m rbbench check-gpu --require A100 > /dev/null
if [ -n "${RELEASED:-}" ]; then
    exec python -m rbbench eval-baseline --released "$RELEASED" --device cuda --cohort tasks/eval_cohort.json \
        --audit "$AUDIT"
fi
if [ -n "${BASELINE:-}" ]; then
    exec python -m rbbench eval-baseline --device cuda --cohort tasks/eval_cohort.json --audit "$AUDIT"
fi
exec python -m rbbench eval-task --manifest "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" \
    --device cuda --cohort "${COHORT:-tasks/eval_cohort.json}" --audit "$AUDIT" --skip-if-complete

#!/bin/bash
# Phase F/I pilot on real data with shortened budgets (experiments/pilot.yaml: 2 / 6 min):
# for every model, train until killed (timeout sends USR1), resume to completion, confirm
# a rerun skips, then evaluate on a few test trajectories. Uses tasks/pilot.jsonl:
#   python -m rbbench make-sweep --experiment pilot --fractions 0.1 --seeds 0 --out tasks/pilot.jsonl
#   sbatch cluster/slurm/pilot_a100.sbatch
#SBATCH --job-name=rb4-pilot
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
# DataLoader workers: all CPUs of the job but one (runtime settings never change a run's identity).
export RBBENCH_NUM_WORKERS=${RBBENCH_NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-5} - 1 ))}
MANIFEST=${MANIFEST:-tasks/pilot.jsonl}
export OUTPUT_ROOT=${PILOT_OUTPUT_ROOT:-$OUTPUT_ROOT/pilot}
AUDIT=${AUDIT:-outputs/audit/dataset_audit.json}

echo "host $(hostname)  job $SLURM_JOB_ID  commit $(git rev-parse HEAD)"
python -m rbbench check-gpu --require A100 > /dev/null
n=$(grep -c . "$MANIFEST")
for ((i = 0; i < n; i++)); do
    echo "=== task $i: interrupted run (USR1 after 150 s)"
    set +e
    timeout -s USR1 150 python -m rbbench train-task --manifest "$MANIFEST" --task-index "$i" \
        --device cuda --resume-if-present --skip-if-complete --allow-dirty
    echo "exit code $? (3 = preempted with state saved)"
    set -e
    echo "=== task $i: resume"
    python -m rbbench train-task --manifest "$MANIFEST" --task-index "$i" \
        --device cuda --resume-if-present --skip-if-complete --allow-dirty
    echo "=== task $i: rerun must skip"
    python -m rbbench train-task --manifest "$MANIFEST" --task-index "$i" \
        --device cuda --resume-if-present --skip-if-complete --allow-dirty | grep already_complete
    python -m rbbench eval-task --manifest "$MANIFEST" --task-index "$i" --device cuda \
        --cohort tasks/eval_cohort.json --audit "$AUDIT" --max-trajectories 4
done
echo "PILOT DONE -> $OUTPUT_ROOT"

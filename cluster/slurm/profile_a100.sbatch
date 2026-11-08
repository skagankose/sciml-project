#!/bin/bash
# Phase G feasibility on real 512x128 data: batch 1 first, then larger batches,
# fp32 and bf16, for all four models. Appends to outputs/pilots/feasibility.csv.
#   sbatch cluster/slurm/profile_a100.sbatch
#SBATCH --job-name=rb4-profile
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
# DataLoader workers: all CPUs of the job but one (runtime settings never change a run's identity).
export RBBENCH_NUM_WORKERS=${RBBENCH_NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-5} - 1 ))}
OUT=${OUT:-outputs/pilots/feasibility.csv}

echo "host $(hostname)  job $SLURM_JOB_ID  commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m rbbench check-gpu --require A100 > /dev/null
python scripts/compare_upstream.py --device cuda --out outputs/pilots/compare_upstream.json
for bs in ${BATCH_SIZES:-1 4 8 16}; do
    python -m rbbench profile --device cuda --batch-size "$bs" --steps 20 --warmup 5 \
        --precision fp32 bf16 --out "$OUT"
done
echo "PROFILE DONE -> $OUT"

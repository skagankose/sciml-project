#!/bin/bash
# Download + verify rayleigh_benard (~367 GB) on a free CPU node. Preemptible: resubmit
# and it resumes partial files and skips verified splits.
#   sbatch cluster/slurm/download_cpu.sbatch             # all splits
#   sbatch --export=ALL,SPLITS="valid test" cluster/slurm/download_cpu.sbatch
#SBATCH --job-name=rb4-download
#SBATCH --partition=free
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
df -h "/pub/$USER" | tail -1
# shellcheck disable=SC2086
bash scripts/download_rb.sh ${SPLITS:-train valid test}

#!/bin/bash
# Download + verify rayleigh_benard (~367 GB) on a GPU node of the class account (paid `gpu`
# partition only, owner 2026-11-21; the GPU is idle, ~34 SU/h; the full download took 1 h 17 min). Resubmit
# and it resumes partial files and skips verified splits.
#   sbatch cluster/slurm/download_cpu.sbatch             # all splits
#   sbatch --export=ALL,SPLITS="valid test" cluster/slurm/download_cpu.sbatch
#SBATCH --job-name=rb4-download
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
df -h "/pub/$USER" | tail -1
# shellcheck disable=SC2086
bash scripts/download_rb.sh ${SPLITS:-train valid test}

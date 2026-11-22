#!/bin/bash
# Build or update the `well-rb` env (cluster/setup_env.sh) on a GPU node of the class account
# (paid `gpu` partition only, owner 2026-11-21; the GPU is idle). Submit from the
# repo root; the log is logs/rb4-setup-<jobid>.out and ends with "done; commit env/hpc3-*".
#   mkdir -p logs && sbatch cluster/slurm/setup_cpu.sbatch
#SBATCH --job-name=rb4-setup
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=5G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
bash cluster/setup_env.sh

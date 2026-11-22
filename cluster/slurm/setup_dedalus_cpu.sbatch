#!/bin/bash
# Build the `rb-dedalus` env (cluster/setup_dedalus.sh) and run the solver self-test.
# Class account, paid `gpu` partition only (owner, 2026-11-21); CPU work, so the GPU it holds is idle.
# The log is logs/rb4-dedalus-setup-<jobid>.out and ends with "passed": true and "done; pull env/...".
#   mkdir -p logs && sbatch cluster/slurm/setup_dedalus_cpu.sbatch
#SBATCH --job-name=rb4-dedalus-setup
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
bash cluster/setup_dedalus.sh

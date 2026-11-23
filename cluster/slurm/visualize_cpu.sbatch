#!/bin/bash
# Frames for the rollout figures (scripts/visualize_rollouts.py): our 10% / 4 h / seed 0 models rolled out on two test
# trajectories (Ra 1e8 and 1e10, Pr 1, from frame 80), plus the true flow. Class account, paid `gpu` partition only (owner, 2026-11-21);
# the job holds a GPU, so the models run on it (VIS_DEVICE=cpu for the CPU).
# Writes outputs/visuals/rollout_*__f<fraction>_<label>_s<seed>.npz (~5 MB each; rbbench.visuals); scripts/findings.py
# draws them on the Mac after cluster/pull.sh.
#   sbatch cluster/slurm/visualize_cpu.sbatch                                          # 10% / 4 h / seed 0
#   sbatch --export=ALL,VIS_ARGS="--fraction 1.0" cluster/slurm/visualize_cpu.sbatch   # 100% / 4 h / seed 0
#SBATCH --job-name=rb4-visualize
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "host $(hostname)  job ${SLURM_JOB_ID}  commit $(git rev-parse HEAD)"
python scripts/visualize_rollouts.py --device "${VIS_DEVICE:-cuda}" ${VIS_ARGS:-}

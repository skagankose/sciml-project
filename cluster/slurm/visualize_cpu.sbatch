#!/bin/bash
# Frames for the rollout figures (scripts/visualize_rollouts.py): our 10% / 4 h / seed 0 models rolled out on two test
# trajectories (Ra 1e8 and 1e10, Pr 1, from frame 80), plus the true flow. CPU is enough; free partition, no SUs.
# Writes outputs/visuals/rollout_*.npz (~5 MB each); scripts/findings.py draws them on the Mac after cluster/pull.sh.
#   sbatch cluster/slurm/visualize_cpu.sbatch
#SBATCH --job-name=rb4-visualize
#SBATCH --partition=free
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "host $(hostname)  job ${SLURM_JOB_ID}  commit $(git rev-parse HEAD)"
python scripts/visualize_rollouts.py --device cpu ${VIS_ARGS:-}

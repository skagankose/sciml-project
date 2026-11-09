#!/bin/bash
# Pipeline check (gate D2): The Well's released CNextU-Net and TFNO checkpoints, one-step VRMSE
# on the valid and test splits through our data path, compared with the published values, plus the
# paper's rollout from frame 0 (Table 3). Each split takes ~80 min (job 59397615); 2 h was too short.
# Evaluation only (AGENTS.md rule 3). Needs verified valid and test downloads. GPU time: ask
# the owner first. Submit from the repo root:
#   mkdir -p logs && sbatch cluster/slurm/released_a100.sbatch
# The log ends with "PIPELINE CHECK PASSED" or "PIPELINE CHECK: MISMATCH"; results are in
# outputs/baselines/released/released_check.json. Any free GPU would do (--gres=gpu:A30:1).
#SBATCH --job-name=rb4-released
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
# The checkpoints (~150 MB) go to node-local disk; released.py checks their pinned sha256.
export HF_HOME=${TMPDIR:-/tmp}/hf-$SLURM_JOB_ID
export PYTHONUNBUFFERED=1   # per-split result lines reach the log as they are printed

echo "host $(hostname)  job $SLURM_JOB_ID  commit $(git rev-parse HEAD)"
python -m rbbench check-gpu > /dev/null
exec python -m rbbench eval-released --device cuda

#!/bin/bash
# Numerical-solver baseline (D25, bounded by D29): Dedalus rollouts from the cohort's start frames, one test trajectory
# per array task (Ra 1e6 / 1e8 / 1e10 at Pr 1, 5 trajectories each = tasks 0-14), starts 80 and 140, one CPU core
# each, free partition. Finished rollouts are kept, so a preempted task resumes when resubmitted. One grid per
# submission (six grids, 30 rollouts each):
#   for RES in 512x128 384x96 256x64 192x48 128x32 64x16; do
#       sbatch --array=0-14 --export=ALL,RES=$RES cluster/slurm/solver_cpu.sbatch
#   done
# The original 35-regime design: --export=ALL,RES=...,DESIGN=all with --array=0-34.
# Then score them in well-rb: sbatch cluster/slurm/solver_score_cpu.sbatch
#SBATCH --job-name=rb4-solver
#SBATCH --partition=free
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV_NAME=rb-dedalus source cluster/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
# One MPI rank on one node: keep UCX (under conda-forge MPICH) off the network cards. On the free nodes with a qedr
# RoCE card (hpc3-14-*, hpc3-15-*) MPI_Init otherwise fails in uct_iface_open(rc_verbs/qedr0) (2026-11-21).
export UCX_TLS=${UCX_TLS:-self,sm,tcp}
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse HEAD)  res ${RES:-512x128}  design ${DESIGN:-bounded}"
python scripts/solver_rollouts.py --resolution "${RES:-512x128}" --design "${DESIGN:-bounded}"

#!/bin/bash
# Numerical-solver baseline (D25, bounded by D29): Dedalus rollouts from the cohort's start frames. Bounded design:
# Ra 1e6 / 1e8 / 1e10 at Pr 1, 5 test trajectories each (solver tasks 0-14), starts 80 and 140. Each rollout runs on
# one CPU core; one job runs the tasks of one grid side by side on its 8 cores, longest (Ra 1e10, tasks 0-4) first.
# Class account on the paid `gpu` partition only (owner, 2026-11-21: no free partitions). The job holds one GPU it
# does not use: 32 + 8 = 40 SU per hour. Finished rollouts are kept and skipped, so a resubmission resumes.
# One array index per grid:
#   0 512x128   1 384x96   2 256x64   3 192x48   4 128x32   5 64x16
#   sbatch --array=0-5 cluster/slurm/solver_cpu.sbatch     # all six grids (~1 h for 512x128, minutes for the rest)
#   sbatch --array=3,5 cluster/slurm/solver_cpu.sbatch     # two grids
# RES=<grid> without --array runs one grid, TASKS="1 2 3" a subset of tasks, DESIGN=all the original 35-regime
# design (tasks 0-34). Each task logs to logs/rb4-solver-<job>_<grid>_t<task>.out; this log lists done / FAILED.
# Then score them in well-rb: sbatch cluster/slurm/solver_score_cpu.sbatch
#SBATCH --job-name=rb4-solver
#SBATCH --partition=gpu
#SBATCH --account=eecs195221_class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV_NAME=rb-dedalus source cluster/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 PYTHONUNBUFFERED=1
# One MPI rank per process on one node: keep UCX (under conda-forge MPICH) off the network cards. On nodes with a qedr
# RoCE card (free nodes hpc3-14-*, hpc3-15-*) MPI_Init otherwise fails in uct_iface_open(rc_verbs/qedr0) (2026-11-21).
export UCX_TLS=${UCX_TLS:-self,sm,tcp}
GRIDS=(512x128 384x96 256x64 192x48 128x32 64x16)
RES=${RES:-${GRIDS[${SLURM_ARRAY_TASK_ID:-0}]}}
DESIGN=${DESIGN:-bounded}
if [ -n "${TASKS:-}" ]; then
    read -r -a TASK_LIST <<< "$TASKS"
elif [ "$DESIGN" = all ]; then
    TASK_LIST=($(seq 0 34))
else
    TASK_LIST=($(seq 0 14))
fi
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
PARALLEL=${SLURM_CPUS_PER_TASK:-1}
export SOLVER_CONCURRENT=$(( PARALLEL < ${#TASK_LIST[@]} ? PARALLEL : ${#TASK_LIST[@]} ))  # recorded with each rollout
echo "host $(hostname)  job ${JOB}_${SLURM_ARRAY_TASK_ID:-}  commit $(git rev-parse HEAD)  res $RES  design $DESIGN" \
     " tasks ${TASK_LIST[*]}  side by side $SOLVER_CONCURRENT"
lscpu | grep -m1 'Model name' || true
run_task() {  # <grid> <design> <job> <task>
    local log="logs/rb4-solver-$3_$1_t$4.out"
    if python scripts/solver_rollouts.py --task-index "$4" --resolution "$1" --design "$2" > "$log" 2>&1; then
        echo "task $4: done  $(grep -E 'start [0-9]+:' "$log" | sed 's/^ *//' | cut -c1-60 | paste -sd ';' -)"
    else
        local code=$?; echo "task $4: FAILED (exit $code), see $log"; return 1
    fi
}
export -f run_task
printf '%s\n' "${TASK_LIST[@]}" | xargs -P "$PARALLEL" -n 1 bash -c 'run_task "$@"' _ "$RES" "$DESIGN" "$JOB"
echo "all tasks done at $(date -u +%FT%TZ)"

#!/bin/bash
# Run a training command and restart it inside the same Slurm allocation after a crash (docs/final_plan.md WP1).
# Use it as the last command of an sbatch script, so that it replaces the batch shell and receives Slurm's signals:
#   exec bash cluster/run_with_restarts.sh python -m rbbench train-task ...
#
# * Exit 0 (done) and 3 (state saved after USR1/SIGTERM: time limit or preemption) end the job with that code.
# * Any other exit is a crash, e.g. a /pub `Errno 70` in a loader worker or in an import (D24). The command runs
#   again and resumes from its last checkpoint, at most MAX_RESTARTS times (default 2), after RESTART_DELAY seconds
#   (default 60), and only while at least MIN_MINUTES_LEFT minutes (default 30) of the allocation remain.
#   Otherwise the job ends with the crash's code and the task is resubmitted as before.
# * USR1 and TERM are forwarded to the command, so the training loop still saves its state before the time limit
#   (--signal=B:USR1@300). After either signal there is no restart.
# * CLEANUP_DIR, if set, is deleted when this script exits (the job's node-local copy of the data).
set -uo pipefail
MAX_RESTARTS=${MAX_RESTARTS:-2}
MIN_MINUTES_LEFT=${MIN_MINUTES_LEFT:-30}
RESTART_DELAY=${RESTART_DELAY:-60}
child=0
stopping=0

cleanup() { if [ -n "${CLEANUP_DIR:-}" ]; then rm -rf "$CLEANUP_DIR"; fi; }
forward() {
    stopping=1
    if [ "$child" -gt 0 ]; then kill -s "$1" "$child" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'forward USR1' USR1
trap 'forward TERM' TERM

minutes_left() {  # minutes until the allocation ends; empty when unknown (outside Slurm)
    if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
        echo $(( (SLURM_JOB_END_TIME - $(date +%s)) / 60 ))
    elif [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue > /dev/null 2>&1; then
        # squeue %L: [days-]hours:minutes:seconds, minutes:seconds, or UNLIMITED
        squeue -h -j "$SLURM_JOB_ID" -o %L 2> /dev/null | awk -F'[-:]' '
            /-/      { print $1 * 1440 + $2 * 60 + $3; exit }
            NF == 3  { print $1 * 60 + $2; exit }
            NF == 2  { print $1; exit }'
    fi
}

attempt=0
while :; do
    "$@" &
    child=$!
    while :; do  # `wait` also returns when a trapped signal arrives; keep waiting for the command itself
        wait "$child"
        rc=$?
        kill -0 "$child" 2> /dev/null || break
    done
    child=0
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$stopping" -eq 1 ]; then
        exit "$rc"
    fi
    if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
        echo "[restart] exit code $rc after $attempt restart(s); giving up" >&2
        exit "$rc"
    fi
    left=$(minutes_left)
    if [ -n "$left" ] && [ "$left" -lt "$MIN_MINUTES_LEFT" ]; then
        echo "[restart] exit code $rc with $left min left (< $MIN_MINUTES_LEFT); not restarting" >&2
        exit "$rc"
    fi
    attempt=$((attempt + 1))
    echo "[restart] exit code $rc at $(date -u +%FT%TZ); restart $attempt/$MAX_RESTARTS in ${RESTART_DELAY} s" >&2
    sleep "$RESTART_DELAY" &
    wait $! || true
    if [ "$stopping" -eq 1 ]; then
        exit "$rc"
    fi
done

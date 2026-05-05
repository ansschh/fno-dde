#!/bin/bash
# Post-sweep waiter — auto-fires the post-sweep batch when training is done.
#
# Polls every 120s for any of these patterns to be running:
#   - dispatcher scripts (_launch_caltech_offload, _launch_full_retrain,
#     _bootstrap_runpod_*, _pod2_recovery, _rescue_waiter)
#   - training (train_apebench_smoke.py)
#   - data generation (gen_dde_pde_data.py)
#   - any prior eval (capture_paper_artifacts, eval_*_dense)
# When NONE of those are running for 3 consecutive checks (~6 min), launches
# scripts/_post_sweep_batch.sh detached and exits.
#
# Pod-side launch (do this once, then forget):
#   setsid nohup bash scripts/_post_sweep_waiter.sh \
#     < /dev/null > train_logs/_post_sweep_waiter_launch.log 2>&1 &
#
# Coexists with rescue_waiter.sh: the rescue waiter re-launches the
# dispatcher on cleanup; the post-sweep waiter waits until the rescued
# cells also finish before firing the batch.

set -uo pipefail
cd /workspace/dde-fno

LOG=train_logs/_post_sweep_waiter.log
mkdir -p train_logs
echo "[post-sweep-waiter] $(date) start" > "$LOG"

PATTERNS=(
    "_launch_caltech_offload\.sh"
    "_launch_full_retrain\.sh"
    "_bootstrap_runpod_offload\.sh"
    "_bootstrap_runpod_orbit\.sh"
    "_pod2_recovery\.sh"
    "_rescue_waiter\.sh"
    "_slot5_fix_and_watch\.sh"
    "train_apebench_smoke\.py"
    "gen_dde_pde_data\.py"
    "capture_paper_artifacts\.py"
    "capture_paper_artifacts_worker\.py"
    "eval_equivariance_dense\.py"
    "eval_adversarial_dense\.py"
    "eval_noise_dense\.py"
)

is_anyone_running() {
    for pat in "${PATTERNS[@]}"; do
        if pgrep -af "$pat" 2>/dev/null | grep -v "_post_sweep_waiter\|grep " > /dev/null; then
            return 0
        fi
    done
    return 1
}

# 60s initial grace so pgrep can observe the just-launched dispatcher.
sleep 60

empty_checks=0
while true; do
    if is_anyone_running; then
        empty_checks=0
        n_running=$(for pat in "${PATTERNS[@]}"; do
            pgrep -af "$pat" 2>/dev/null | grep -v "_post_sweep_waiter\|grep "
        done | wc -l)
        echo "[post-sweep-waiter] $(date) busy ($n_running procs)" >> "$LOG"
    else
        empty_checks=$((empty_checks + 1))
        echo "[post-sweep-waiter] $(date) idle (consecutive=$empty_checks)" >> "$LOG"
        if [ "$empty_checks" -ge 3 ]; then
            break
        fi
    fi
    sleep 120
done

echo "[post-sweep-waiter] $(date) sweep idle for 3 checks, launching batch" >> "$LOG"

if [ ! -f scripts/_post_sweep_batch.sh ]; then
    echo "[post-sweep-waiter] $(date) FATAL: scripts/_post_sweep_batch.sh missing!" >> "$LOG"
    echo "[post-sweep-waiter] $(date) Likely cause: deploy not yet complete on this pod." >> "$LOG"
    exit 1
fi

setsid nohup bash scripts/_post_sweep_batch.sh \
    < /dev/null \
    > train_logs/_post_sweep_batch_launch.log 2>&1 &
BATCH_PID=$!
echo "[post-sweep-waiter] $(date) batch PID $BATCH_PID launched" >> "$LOG"
sleep 5
if kill -0 "$BATCH_PID" 2>/dev/null; then
    echo "[post-sweep-waiter] $(date) batch confirmed alive" >> "$LOG"
else
    echo "[post-sweep-waiter] $(date) WARNING: batch PID not alive 5s after launch" >> "$LOG"
fi

echo "[post-sweep-waiter] $(date) waiter exiting (batch is detached and running)" >> "$LOG"

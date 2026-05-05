#!/bin/bash
# A-fix LEMO-PC retrain launcher: 45 cells across 8 GPUs with 24 workers.
#
# Each worker grabs the next available cell index via flock-protected
# counter, runs train_apebench_smoke.py with that cell's args, writes
# log to train_logs/a_fix/cell_<idx>.log.
#
# Usage on pod:
#   cd /workspace/dde-fno && bash scripts/_a_fix_launcher.sh

set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

N_GPU=${N_GPU:-8}
N_WORKERS=${N_WORKERS:-24}        # 3 cells per GPU
LOG_DIR=$REPO/train_logs/a_fix
COUNTER_FILE=$REPO/train_logs/a_fix/.next_idx
mkdir -p "$LOG_DIR"

TOTAL=$(python3 -c "from scripts._a_fix_cells import all_cells; print(len(all_cells()))")
echo "[a_fix] total cells: $TOTAL, workers: $N_WORKERS, GPUs: $N_GPU"

echo 0 > "$COUNTER_FILE"

worker() {
    local worker_id=$1
    local gpu=$(( worker_id % N_GPU ))
    while true; do
        # flock-protected counter increment
        local idx
        idx=$(
            (
                flock -x 200
                idx=$(cat "$COUNTER_FILE")
                echo $((idx + 1)) > "$COUNTER_FILE"
                echo "$idx"
            ) 200>"$COUNTER_FILE.lock"
        )
        if [ "$idx" -ge "$TOTAL" ]; then
            return 0
        fi
        local log="$LOG_DIR/cell_$(printf '%03d' $idx).log"
        echo "[w$worker_id gpu$gpu] cell $idx -> $log" >&2
        CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/_run_a_fix_cell.py "$idx" "$gpu" \
            > "$log" 2>&1
    done
}

for w in $(seq 0 $((N_WORKERS - 1))); do
    worker $w &
done
wait
echo "[a_fix] all workers done"

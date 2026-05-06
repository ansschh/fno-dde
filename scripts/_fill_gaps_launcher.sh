#!/bin/bash
# Fill-table-gaps launcher: 84 cells across 8 GPUs / 24 workers.
# Pattern mirrors _a_fix_launcher.sh.
#
# Usage on pod:
#   cd /workspace/dde-fno && bash scripts/_fill_gaps_launcher.sh

set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

N_GPU=${N_GPU:-8}
N_WORKERS=${N_WORKERS:-24}
LOG_DIR=$REPO/train_logs/fill_gaps
COUNTER_FILE=$REPO/train_logs/fill_gaps/.next_idx
mkdir -p "$LOG_DIR"

TOTAL=$(python3 -c "from scripts._fill_table_gaps_cells import all_cells; print(len(all_cells()))")
echo "[fill_gaps] total cells: $TOTAL, workers: $N_WORKERS, GPUs: $N_GPU"

echo 0 > "$COUNTER_FILE"

worker() {
    local worker_id=$1
    local gpu=$(( worker_id % N_GPU ))
    while true; do
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
        CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/_run_fill_gaps_cell.py "$idx" "$gpu" \
            > "$log" 2>&1
    done
}

for w in $(seq 0 $((N_WORKERS - 1))); do
    worker $w &
done
wait
echo "[fill_gaps] all workers done"

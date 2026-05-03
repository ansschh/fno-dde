#!/bin/bash
# B5 + P3 dispatcher — runs cells from _b5_p3_cells.py across N GPUs.
#
# Usage: bash scripts/_launch_b5_p3.sh <START> <COUNT> <NGPU>
#
# Example:  bash scripts/_launch_b5_p3.sh 0 40 8   # all 40 cells on 8 GPUs
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-40}
NGPU=${3:-8}

mkdir -p train_logs/b5_p3

NCELLS=$COUNT
echo "[b5_p3] $NCELLS cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

# Generate cell list as JSON once
python3 scripts/_b5_p3_cells.py --json > /tmp/b5_p3_cells.json

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((START + COUNT - 1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      log=train_logs/b5_p3/cell_${i}.log
      echo "[GPU $gpu] cell $i -> $log"
      python3 -u scripts/_run_b5_p3_cell.py $i $gpu \
        > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $i FAILED rc=$? (continuing)"
      echo "[GPU $gpu] done cell $i"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[b5_p3] all $NCELLS cells (indices $START..$((START+COUNT-1))) complete."

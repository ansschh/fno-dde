#!/bin/bash
# Per-pod launcher for the Caltech offload.  Bash dispatches the python
# per-cell runner; python handles the subprocess + arg-list cleanly.
#
# Usage: bash _launch_caltech_offload.sh <START> <COUNT> <NGPU>
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-50}
NGPU=${3:-8}

mkdir -p train_logs/offload

NCELLS=$COUNT
echo "[launch] $NCELLS cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((START + COUNT - 1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      log=train_logs/offload/cell_${i}.log
      echo "[GPU $gpu] cell $i -> $log"
      python3 -u scripts/_run_offload_cell.py $i $gpu \
        > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $i FAILED rc=$? (continuing)"
      echo "[GPU $gpu] done cell $i"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[launch] all $NCELLS cells (indices $START..$((START+COUNT-1))) complete."

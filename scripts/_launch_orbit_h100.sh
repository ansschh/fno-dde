#!/bin/bash
# Orbit OOD parallel dispatcher for H100 pod.
# Runs all 27 cells from _orbit_h100_cells.py across 8 GPUs concurrently.
#
# Usage: bash scripts/_launch_orbit_h100.sh <START> <COUNT> <NGPU>
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-27}
NGPU=${3:-8}

mkdir -p train_logs/orbit_h100

NCELLS=$COUNT
echo "[orbit_h100] $NCELLS cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((START + COUNT - 1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      log=train_logs/orbit_h100/cell_${i}.log
      echo "[GPU $gpu] cell $i -> $log"
      python3 -u scripts/_run_orbit_h100_cell.py $i $gpu \
        > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $i FAILED rc=$? (continuing)"
      echo "[GPU $gpu] done cell $i"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[orbit_h100] all $NCELLS cells (indices $START..$((START+COUNT-1))) complete."

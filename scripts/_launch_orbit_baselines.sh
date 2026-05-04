#!/bin/bash
# Orbit OOD baselines parallel dispatcher for 16x H100 pod.
# Runs all 90 cells from _orbit_baselines_cells.py across N GPUs.
#
# Usage: bash scripts/_launch_orbit_baselines.sh <START> <COUNT> <NGPU>
# Example for 16-H100 pod: bash scripts/_launch_orbit_baselines.sh 0 90 16
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-90}
NGPU=${3:-16}

mkdir -p train_logs/orbit_baselines

NCELLS=$COUNT
echo "[orbit_baselines] $NCELLS cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((START + COUNT - 1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      log=train_logs/orbit_baselines/cell_${i}.log
      echo "[GPU $gpu] cell $i -> $log"
      python3 -u scripts/_run_orbit_baselines_cell.py $i $gpu \
        > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $i FAILED rc=$? (continuing)"
      echo "[GPU $gpu] done cell $i"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[orbit_baselines] all $NCELLS cells (indices $START..$((START+COUNT-1))) complete."

#!/bin/bash
# FNO+FiLM orbit OOD dispatcher (18 cells across N GPUs).
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-18}
NGPU=${3:-8}

mkdir -p train_logs/orbit_fnofilm
echo "[fnofilm] $COUNT cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((START + COUNT - 1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      log=train_logs/orbit_fnofilm/cell_${i}.log
      echo "[GPU $gpu] cell $i -> $log"
      python3 -u scripts/_run_orbit_fnofilm_cell.py $i $gpu \
        > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $i FAILED rc=$? (continuing)"
      echo "[GPU $gpu] done cell $i"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[fnofilm] complete."

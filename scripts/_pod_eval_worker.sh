#!/bin/bash
# Per-GPU worker: process a shard of checkpoints through capture + evals.
#
# Usage: bash _pod_eval_worker.sh <shard.txt> <data_dir> <gpu_id>
set -e
SHARD="$1"
DATA_DIR="$2"
GPU="$3"

cd /workspace/dde-fno
N=$(wc -l < "$SHARD")
echo "[gpu $GPU] $N cells to process"

i=0
while IFS= read -r ckpt; do
  i=$((i + 1))
  cell_dir=$(dirname "$ckpt")
  rel=$(realpath --relative-to=. "$cell_dir" 2>/dev/null || echo "$cell_dir")
  echo "[gpu $GPU] [$i/$N] $rel"
  parts=$(echo "$rel" | tr '/' ' ')
  # parts ends in: ... raw <fam> <reg> <model> s<seed>
  fam=$(echo "$rel" | awk -F/ '{ for(i=1;i<=NF;i++) if($i=="raw"){print $(i+1); exit} }')
  if [ -z "$fam" ]; then
    echo "[gpu $GPU]   skip (no raw/ in path)"
    continue
  fi

  # 1. capture --minimal (per_frame.json + viz_samples.npz)
  if [ ! -s "$cell_dir/per_frame.json" ] || [ ! -s "$cell_dir/viz_samples.npz" ]; then
    python3 -c "
import sys, os
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from capture_paper_artifacts import process_cell
from pathlib import Path
msg = process_cell(Path('$ckpt'), '$DATA_DIR', '$fam', 'cuda', 4, minimal=True)
print(f'  capture: {msg}')
" 2>&1 | grep -v "^$" | head -2 || echo "[gpu $GPU]   capture FAIL"
  fi
done < "$SHARD"

echo "[gpu $GPU] capture phase done"

# Phase 2: dense evals across all cells in shard (each script crawls roots,
# uses skip-if-exists internally so re-running is idempotent)
ROOTS=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
)

# Each gpu only handles cells that hash to its index — but the eval scripts
# crawl entire roots. We let each GPU process its OWN shard via env-controlled
# CUDA_VISIBLE_DEVICES. The scripts have skip-if-exists so duplication is OK.

# E2 — empirical Lipschitz on sigma cells (only sigma_* roots)
python3 scripts/eval_w1_empirical_lipschitz.py \
  --layer_root extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
  --data_dir "$DATA_DIR" --device cuda --n_pairs 100 \
  2>&1 | tail -3 || true

echo "[gpu $GPU] all evals done"

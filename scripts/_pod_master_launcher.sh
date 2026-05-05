#!/bin/bash
# Master launcher: shard cells and dispatch to 8 GPU workers.
#
# Usage:
#   bash scripts/_pod_master_launcher.sh [steps]
#   steps default: capture,lipschitz,equivariance,per_frame
set -e
cd /workspace/dde-fno

STEPS="${1:-capture,lipschitz,equivariance,per_frame}"
N_GPUS="${N_GPUS:-8}"
DATA_DIR="data_dde_pde"

ROOTS=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/film_ablation_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod
  extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_runpod
  extracted/pod_pulls_2026_05_03_final/227/outputs
  extracted/pod_pulls_2026_05_03_final/A40/outputs
  extracted/pod_pulls_2026_05_03_final/244/outputs
)

mkdir -p logs/pod_evals
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Collect all checkpoints
ALL=$TMPDIR/all_ckpts.txt
> "$ALL"
for r in "${ROOTS[@]}"; do
  if [ -d "$r" ]; then
    find "$r" -name "best_model.pt" >> "$ALL"
  fi
done
TOTAL=$(wc -l < "$ALL")
echo "[master] $TOTAL checkpoints across ${#ROOTS[@]} roots, steps=$STEPS"
if [ "$TOTAL" -eq 0 ]; then
  echo "no checkpoints found"; exit 1
fi

# Shard
for g in $(seq 0 $((N_GPUS - 1))); do
  awk -v g=$g -v n=$N_GPUS 'NR%n==g' "$ALL" > "$TMPDIR/shard_$g.txt"
  n=$(wc -l < "$TMPDIR/shard_$g.txt")
  echo "  gpu $g: $n cells"
done

# Launch workers
PIDS=()
for g in $(seq 0 $((N_GPUS - 1))); do
  shard="$TMPDIR/shard_$g.txt"
  if [ ! -s "$shard" ]; then continue; fi
  cp "$shard" "logs/pod_evals/shard_${g}.txt"
  log="logs/pod_evals/gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 -u scripts/_pod_unified_worker.py \
      --shard "logs/pod_evals/shard_${g}.txt" \
      --data_dir "$DATA_DIR" \
      --gpu $g \
      --steps "$STEPS" \
    > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[master] launched gpu $g (pid $!)"
done

# Wait
echo "[master] ${#PIDS[@]} workers running. Waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  worker pid $pid exit nonzero"
done

echo "=== master complete ==="
for g in $(seq 0 $((N_GPUS - 1))); do
  log="logs/pod_evals/gpu${g}.log"
  if [ -f "$log" ]; then
    n_ok=$(grep -c "ok " "$log" 2>/dev/null || true)
    n_skip=$(grep -c "skip" "$log" 2>/dev/null || true)
    n_fail=$(grep -c "FAIL" "$log" 2>/dev/null || true)
    echo "  gpu $g: ok=$n_ok skip=$n_skip fail=$n_fail"
  fi
done

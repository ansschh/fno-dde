#!/bin/bash
# Pod-side launcher: run all post-hoc evals across 8 GPUs in parallel.
#
# Per-GPU shard runs sequentially through:
#   1. capture_paper_artifacts.py --minimal       (per_frame.json + viz_samples.npz)
#   2. eval_w1_empirical_lipschitz.py             (W1-E2)
#   3. eval_per_frame_dense.py                    (F06, F_boundary)
#   4. eval_equivariance_dense.py                 (F08)
#   5. eval_adversarial_dense.py                  (F11 left)
#   6. eval_noise_dense.py                        (F11 right)
#
# Usage:
#   bash scripts/_pod_run_all_evals.sh
set -e
cd /workspace/dde-fno

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

DATA_DIR=data_dde_pde
N_GPUS=${N_GPUS:-8}

mkdir -p logs/pod_evals

# Collect all checkpoints from all roots
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
ALL_CKPTS=$TMPDIR/all_ckpts.txt
> "$ALL_CKPTS"
for root in "${ROOTS[@]}"; do
  if [ -d "$root" ]; then
    find "$root" -name "best_model.pt" >> "$ALL_CKPTS"
  fi
done
TOTAL=$(wc -l < "$ALL_CKPTS")
echo "[pod-evals] $TOTAL checkpoints across ${#ROOTS[@]} sweep roots"

if [ "$TOTAL" -eq 0 ]; then
  echo "no checkpoints found; check ROOTS"
  exit 1
fi

# Shard checkpoints across GPUs
for g in $(seq 0 $((N_GPUS - 1))); do
  awk -v g=$g -v n=$N_GPUS 'NR%n==g' "$ALL_CKPTS" > "$TMPDIR/shard_$g.txt"
  n=$(wc -l < "$TMPDIR/shard_$g.txt")
  echo "  gpu $g: $n cells"
done

# Run per-GPU worker in parallel.
PIDS=()
for g in $(seq 0 $((N_GPUS - 1))); do
  shard="$TMPDIR/shard_$g.txt"
  if [ ! -s "$shard" ]; then continue; fi
  log="logs/pod_evals/gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    bash scripts/_pod_eval_worker.sh "$shard" "$DATA_DIR" "$g" \
    > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[pod-evals] launched GPU $g worker (PID $!)"
done

echo "[pod-evals] ${#PIDS[@]} workers running. Waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  worker $pid exited non-zero"
done

echo "=== pod-evals complete ==="
for g in $(seq 0 $((N_GPUS - 1))); do
  log="logs/pod_evals/gpu${g}.log"
  if [ -f "$log" ]; then
    n_done=$(grep -c "ok " "$log" 2>/dev/null || echo 0)
    n_skip=$(grep -c "skip" "$log" 2>/dev/null || echo 0)
    n_fail=$(grep -c "FAIL\|fail\|Failed" "$log" 2>/dev/null || echo 0)
    echo "  gpu $g: ok=$n_done skip=$n_skip fail=$n_fail"
  fi
done

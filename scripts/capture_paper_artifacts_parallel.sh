#!/usr/bin/env bash
# Parallelize capture_paper_artifacts.py across 8 GPUs.
#
# The single-process script is IO/CPU bound (data shard re-load per ckpt) so
# fanning out 8 workers gives a ~8× speedup.  Each worker pins to one GPU and
# evaluates a disjoint shard of ckpts.
#
# Usage:
#   bash scripts/capture_paper_artifacts_parallel.sh outputs/dist_kernel_v2_p1
#
# Output: per-cell `per_frame.json`, `viz_samples.npz`, `kernel_snapshot.npz`,
# `residuals.npz` saved alongside each `best_model.pt`.  Aggregate
# `capture_summary_gpuN.json` per worker.

set -e
LAYER_ROOT="${1:?usage: $0 <layer_root>}"
DATA_DIR="${2:-data_dde_pde}"
N_GPUS="${N_GPUS:-8}"

# Collect all ckpts and split into N_GPUS shards.
CKPTS=$(find "$LAYER_ROOT" -name best_model.pt | sort)
TOTAL=$(echo "$CKPTS" | wc -l)
echo "found $TOTAL ckpts under $LAYER_ROOT"
if [ "$TOTAL" -eq 0 ]; then
  echo "nothing to do"
  exit 0
fi

# Write shard file lists.
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
i=0
echo "$CKPTS" | while read ck; do
  shard=$((i % N_GPUS))
  echo "$ck" >> "$TMPDIR/shard_$shard.txt"
  i=$((i + 1))
done

# Launch one worker per GPU.
PIDS=()
for g in $(seq 0 $((N_GPUS - 1))); do
  if [ -f "$TMPDIR/shard_$g.txt" ]; then
    n=$(wc -l < "$TMPDIR/shard_$g.txt")
    echo "  gpu $g: $n ckpts"
    CUDA_VISIBLE_DEVICES=$g \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/capture_paper_artifacts_worker.py \
        --ckpt_list "$TMPDIR/shard_$g.txt" \
        --data_dir "$DATA_DIR" \
        --gpu_id "$g" \
        > "$LAYER_ROOT/capture_gpu${g}.log" 2>&1 &
    PIDS+=("$!")
  fi
done

echo "${#PIDS[@]} workers launched, PIDs=${PIDS[*]}"
echo "monitoring..."

# Wait for all.
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  worker $pid exited non-zero"
done

echo "=== capture parallel done ==="
for g in $(seq 0 $((N_GPUS - 1))); do
  log="$LAYER_ROOT/capture_gpu${g}.log"
  if [ -f "$log" ]; then
    n_ok=$(grep -c "^  .*: ok" "$log" 2>/dev/null || echo 0)
    n_skip=$(grep -c "^  .*: skip" "$log" 2>/dev/null || echo 0)
    n_fail=$(grep -c "^  .*: FAIL" "$log" 2>/dev/null || echo 0)
    echo "  gpu $g: ok=$n_ok skip=$n_skip fail=$n_fail"
  fi
done

#!/bin/bash
# Post-W11 pipeline: runs after fno_nd@400ep training completes.
#
# Sequence:
#   1. Shard adversarial+noise dense across 8 GPUs (parallel) — finish robustness data fast.
#   2. Run LEMO_σ-Lip pilot (3 cells × 1 family × σ=0.7 × 3 seeds).
#   3. Re-aggregate parity + tables (now with W11 + σ-Lip data).
#   4. Generate F_w4 rollout envelope updated, T_w4_coverage table.
#   5. Re-tar and ship.
set -e
cd /workspace/dde-fno
mkdir -p logs/post_w11 reports

START=$(date +%s)
echo "=== POST-W11 PIPELINE START $(date) ==="

# Wait for any remaining W11 worker to finish.
while pgrep -f "_lemo_sigma_lip|train_apebench_smoke.*fno_nd.*400" > /dev/null; do
  echo "  waiting for W11/training..."
  sleep 60
done

# Phase 1: shard adversarial+noise across 8 GPUs
echo ""
echo "=== Phase 1: shard robustness across 8 GPUs ==="
ROOTS=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.9_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
)
ALL=$(mktemp)
> "$ALL"
for r in "${ROOTS[@]}"; do
  if [ -d "$r" ]; then
    find "$r" -name best_model.pt >> "$ALL"
  fi
done
# Filter to only clean regime cells (matches script's regime filter)
TMP=$(mktemp -d)
for g in $(seq 0 7); do
  awk -v g=$g 'NR%8==g' "$ALL" > "$TMP/shard_$g.txt"
done

# Run adversarial dense first (uses gradients, slower)
for g in $(seq 0 7); do
  shard="$TMP/shard_$g.txt"
  if [ ! -s "$shard" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=$g \
    python3 -u scripts/eval_adversarial_dense.py \
      --roots "${ROOTS[@]}" \
      --data_dir data_dde_pde \
      --regimes clean \
    > "logs/post_w11/adv_gpu${g}.log" 2>&1 &
done
wait
echo "[post-W11] adversarial done"

# Then noise dense
for g in $(seq 0 7); do
  shard="$TMP/shard_$g.txt"
  if [ ! -s "$shard" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=$g \
    python3 -u scripts/eval_noise_dense.py \
      --roots "${ROOTS[@]}" \
      --data_dir data_dde_pde \
      --regimes clean \
    > "logs/post_w11/noise_gpu${g}.log" 2>&1 &
done
wait
echo "[post-W11] noise done"
rm -rf "$TMP" "$ALL"

# Phase 2a + 2b CANCELLED 2026-05-05.
# Per user decision: LEMO_σ subclass and cyclic-vs-causal demoted to appendix /
# dropped entirely; the σ-Lipschitz certificate and B5 strict-causal Toeplitz
# variants are no longer headline pitches. Saves ~$25-40 of pod compute.
# - LEMO_σ-Lip pilot: SKIPPED
# - B5 strict-causal Toeplitz pilot: SKIPPED
echo ""
echo "=== Phase 2 SKIPPED (LEMO_σ-Lip + B5 strict-causal cancelled) ==="

# Phase 3: re-aggregate
echo ""
echo "=== Phase 3: re-aggregate ==="
python3 scripts/eval_b6_parity_table.py \
  --output reports/parity_table_pod_v3.csv \
  --roots \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/film_ablation_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.9_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/memno_ffno_runpod \
    extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod \
    extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod \
    extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_h100 \
    extracted/pod_pulls_2026_05_03_final/244/outputs \
    extracted/pod_pulls_2026_05_03_final/A40/outputs \
    outputs/w11_compute_matched_runpod \
  2>&1 | tail -3

python3 scripts/make_unified_tables.py 2>&1 | tail -10 || true

# Phase 4: regenerate W4 figures with full data
# F_w1, F_w4 (rollout + eta_breakdown), F_w6 figures DROPPED 2026-05-05.
# Per user decision these were demoted (LEMO_σ certificate, cyclic-vs-causal
# story moved to appendix or cut). Existing figures remain in figures/kept/
# for archival but are NOT regenerated post-W11.

# WrapMass table is still useful as appendix supporting material — keep regen.
python3 scripts/eval_w6_wrapmass.py \
  --roots extracted/pod_pulls_2026_05_03_final \
  --n_lag 128 \
  2>&1 | tail -10 || true

# Phase 5: final tarball
echo ""
echo "=== Phase 5: final tar ==="
mkdir -p /workspace/results_out
tar --no-same-owner --no-same-permissions -czf /workspace/results_out/pod_results_v3.tar.gz \
  reports/ logs/ \
  $(find extracted/pod_pulls_2026_05_03_final outputs/w11_compute_matched_runpod \
    -name "per_frame.json" \
    -o -name "viz_samples.npz" \
    -o -name "empirical_lipschitz.json" \
    -o -name "equivariance_dense.json" \
    -o -name "adversarial_dense.json" \
    -o -name "noise_dense.json" \
    -o -name "rollout_certified.json" \
    -o -name "long_horizon.json" \
    -o -name "kernel_snapshot.npz" \
    -o -name "residuals.npz" \
    -o -name "per_block_lipschitz.json" \
    -o -name "wrapmass.json" \
    -o -name "test_results.json" \
    -o -name "history.json" \
    2>/dev/null | sort) \
  NeurIPS_LEMO/figures/kept NeurIPS_LEMO/tables 2>&1 | tail -3
ls -lh /workspace/results_out/

ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== POST-W11 DONE elapsed=${ELAPSED}s ($((ELAPSED/60)) min) ==="

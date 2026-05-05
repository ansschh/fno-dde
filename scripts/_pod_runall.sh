#!/bin/bash
# Final aggregation pipeline — runs after Phase 1 master_run1 completes.
#
# Sequence:
#   1. Run 2 of master launcher (drains T2 cells from later transfer)
#   2. Long-horizon eval (8-way parallel)
#   3. W11 compute-matched FNO training (8-way parallel, ~2.5h)
#   4. Aggregations: parity table, paired_permutation, T01/T02 update
#   5. Figures: F_w1 certified region, F_w3 covering radius, T_lag_modes
#   6. tar all results to /workspace/results_out/
set -e
cd /workspace/dde-fno
mkdir -p logs/pipeline_full reports

START=$(date +%s)
echo "=== POD RUN-ALL PIPELINE START $(date) ==="

# Phase 2: Run 2 of master (T2 cells)
echo ""
echo "=== Phase 2: master pipeline run 2 (drain T2 cells) ==="
bash scripts/_pod_master_launcher.sh capture,lipschitz,equivariance,per_frame \
  2>&1 | tee logs/pipeline_full/phase2_master_run2.log

# Phase 3: long-horizon eval
echo ""
echo "=== Phase 3: long-horizon rollout (h ∈ {64,128,256,512}) ==="
ALL_CKPTS=$(mktemp)
for r in extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_*_runpod \
         extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod \
         extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod \
         extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod; do
  if [ -d "$r" ]; then
    find "$r" -name best_model.pt >> "$ALL_CKPTS"
  fi
done
N_LH=$(wc -l < "$ALL_CKPTS")
echo "  long-horizon on $N_LH ckpts"
TMP_LH=$(mktemp -d)
for g in $(seq 0 7); do
  awk -v g=$g 'NR%8==g' "$ALL_CKPTS" > "$TMP_LH/shard_$g.txt"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 -u scripts/eval_long_horizon.py \
      --shard "$TMP_LH/shard_$g.txt" \
      --data_dir data_dde_pde \
      --n_chain_max 8 \
      --device cuda \
    > "logs/pipeline_full/phase3_lh_gpu${g}.log" 2>&1 &
done
wait
echo "  long-horizon done"
rm -rf "$TMP_LH" "$ALL_CKPTS"

# Phase 4: W11 compute-matched FNO training
echo ""
echo "=== Phase 4: W11 compute-matched FNO@400ep ==="
bash scripts/_w11_launcher.sh 2>&1 | tee logs/pipeline_full/phase4_w11.log

# Phase 5: aggregations
echo ""
echo "=== Phase 5: aggregations ==="
ROOTS_ALL=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/film_ablation_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod
  extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_runpod
  extracted/pod_pulls_2026_05_03_final/244/outputs
  extracted/pod_pulls_2026_05_03_final/A40/outputs
  outputs/w11_compute_matched_runpod
)
python3 scripts/eval_b6_parity_table.py \
  --output reports/parity_table_pod_full.csv \
  --roots "${ROOTS_ALL[@]}" 2>&1 | tail -5

python3 scripts/eval_w1_rollout_certified.py \
  --layer_root extracted/pod_pulls_2026_05_03_final \
  --summary_csv reports/w1_e3_rollout_summary.csv 2>&1 | tail -5

python3 scripts/make_T_lag_modes_ablation.py 2>&1 | tail -5 || true

# Phase 6: figures
echo ""
echo "=== Phase 6: figures ==="
mkdir -p NeurIPS_LEMO/figures/kept/main NeurIPS_LEMO/figures/kept/png
python3 scripts/make_F_w1_certified_region.py \
  --roots extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
          extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod \
          extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod \
  2>&1 | tail -5 || true

python3 scripts/make_F_w3_covering_radius.py \
  --roots extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_runpod \
  2>&1 | tail -10 || true

# Phase 7: tar all artifacts back
echo ""
echo "=== Phase 7: tarring results ==="
mkdir -p /workspace/results_out
tar --no-same-owner --no-same-permissions -czf /workspace/results_out/pod_results_full.tar.gz \
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
    -o -name "test_results.json" \
    -o -name "history.json" \
    2>/dev/null | sort) \
  NeurIPS_LEMO/ 2>&1 | tail -3

ls -lh /workspace/results_out/
ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== POD RUN-ALL DONE elapsed=${ELAPSED}s ($((ELAPSED/60)) min) ==="

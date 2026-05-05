#!/bin/bash
# Full pod pipeline:
#   1. Run master launcher (capture + W1-E2 + equivariance_dense across 8 GPUs)
#   2. Run W1-E3 (rollout vs certified, post-processing)
#   3. Run dense adversarial + noise (8-way)
#   4. Generate W1 figures + lag_modes table
#   5. Tar all results for SCP back
set -e
cd /workspace/dde-fno
mkdir -p logs/pod_pipeline reports

START=$(date +%s)
echo "=== POD PIPELINE START $(date) ==="

# Phase 1: capture + W1-E2 + equivariance dense in parallel across 8 GPUs
echo ""
echo "=== Phase 1: capture + lipschitz + equivariance (8-way) ==="
bash scripts/_pod_master_launcher.sh capture,lipschitz,equivariance \
  2>&1 | tee logs/pod_pipeline/phase1.log

# Phase 2: W1-E3 (post-processing using per_frame.json from Phase 1)
echo ""
echo "=== Phase 2: W1-E3 rollout vs certified envelope ==="
python3 scripts/eval_w1_rollout_certified.py \
  --layer_root extracted/pod_pulls_2026_05_03_final \
  --summary_csv reports/w1_e3_rollout_summary.csv \
  2>&1 | tee logs/pod_pipeline/phase2_e3.log | tail -10

# Phase 3: adversarial + noise dense (sequentially per script, 8-way internally)
echo ""
echo "=== Phase 3: adversarial + noise dense ==="

# eval_adversarial_dense.py / eval_noise_dense.py crawl roots and have their own
# skip-if-exists. Run on whichever subset of cells we want (sigma + film_ablation
# + memory_aware are the most paper-relevant).
ROOTS_PRIMARY=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
)

for script in eval_adversarial_dense.py eval_noise_dense.py; do
  echo "  running $script..."
  python3 scripts/$script \
    --roots "${ROOTS_PRIMARY[@]}" \
    --data_dir data_dde_pde \
    --regimes clean \
    2>&1 | tee logs/pod_pipeline/phase3_$script.log | tail -5
done

# Phase 4: aggregations
echo ""
echo "=== Phase 4: aggregations ==="
python3 scripts/eval_b6_parity_table.py \
  --output reports/parity_table_pod.csv \
  --roots "${ROOTS_PRIMARY[@]}" \
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/film_ablation_runpod \
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/memno_ffno_runpod \
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod \
  extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_runpod \
  extracted/pod_pulls_2026_05_03_final/227/outputs \
  extracted/pod_pulls_2026_05_03_final/A40/outputs \
  extracted/pod_pulls_2026_05_03_final/244/outputs \
  2>&1 | tee logs/pod_pipeline/phase4_parity.log | tail -3

python3 scripts/make_T_lag_modes_ablation.py 2>&1 | tail -10 || true

# Phase 5: W1 figure
echo ""
echo "=== Phase 5: W1 figure ==="
mkdir -p NeurIPS_LEMO/figures/kept/main NeurIPS_LEMO/figures/kept/png
python3 scripts/make_F_w1_certified_region.py \
  --roots extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
          extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod \
          extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod \
  --out_name F_w1_certified_region 2>&1 | tail -10 || true

# Phase 6: tar results
echo ""
echo "=== Phase 6: tarring results ==="
mkdir -p /workspace/results_out
tar --no-same-owner --no-same-permissions \
  -czf /workspace/results_out/pod_results.tar.gz \
  reports/ \
  logs/ \
  $(find extracted/pod_pulls_2026_05_03_final -name "per_frame.json" \
                                                -o -name "viz_samples.npz" \
                                                -o -name "empirical_lipschitz.json" \
                                                -o -name "equivariance_dense.json" \
                                                -o -name "adversarial_dense.json" \
                                                -o -name "noise_dense.json" \
                                                -o -name "rollout_certified.json" \
                                                -o -name "kernel_snapshot.npz" \
                                                -o -name "residuals.npz" \
                                                -o -name "long_rollout.npz" \
                                                -o -name "fft_residual.npz" \
                                                -o -name "equivariance.json" \
                                                2>/dev/null | sort) \
  2>&1 | tail -3 || true
ls -lh /workspace/results_out/pod_results.tar.gz

ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== POD PIPELINE DONE elapsed=${ELAPSED}s ($((ELAPSED/60)) min) ==="

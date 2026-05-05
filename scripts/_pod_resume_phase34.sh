#!/bin/bash
# Resume Phase 3 (long-horizon) and Phase 4 (W11) after the first runall
# crashed both. Then re-run aggregations + figures.
set -e
cd /workspace/dde-fno
mkdir -p logs/resume reports

START=$(date +%s)
echo "=== POD RESUME PHASE 3+4 START $(date) ==="

# Phase 3 retry: long-horizon eval
echo ""
echo "=== Phase 3 retry: long-horizon rollout ==="
ALL_CKPTS=$(mktemp)
for r in extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
         extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod \
         extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod \
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
    > "logs/resume/lh_gpu${g}.log" 2>&1 &
done
wait
echo "  long-horizon done"
rm -rf "$TMP_LH" "$ALL_CKPTS"
echo ""
echo "  long_horizon.json count:"
find /workspace/dde-fno/extracted -name long_horizon.json 2>/dev/null | wc -l

# Phase 4 retry: W11
echo ""
echo "=== Phase 4 retry: W11 compute-matched FNO@400ep ==="
bash scripts/_w11_launcher.sh 2>&1 | tee logs/resume/w11.log | tail -20

# Phase 5 redo: aggregations including W11
echo ""
echo "=== Phase 5 redo: aggregations ==="
python3 scripts/eval_b6_parity_table.py \
  --output reports/parity_table_pod_full_v2.csv \
  --roots \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/film_ablation_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/memno_ffno_runpod \
    extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod \
    extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod \
    extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/p3_sensitivity_runpod \
    extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_runpod \
    extracted/pod_pulls_2026_05_03_final/244/outputs \
    extracted/pod_pulls_2026_05_03_final/A40/outputs \
    outputs/w11_compute_matched_runpod \
  2>&1 | tail -3

# Phase 7 retar
echo ""
echo "=== Phase 7 retar: tar all results ==="
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
  NeurIPS_LEMO/figures/kept/ NeurIPS_LEMO/tables/T_lag_modes_ablation.tex \
  2>&1 | tail -3
ls -lh /workspace/results_out/

ELAPSED=$(($(date +%s) - START))
echo ""
echo "=== POD RESUME DONE elapsed=${ELAPSED}s ($((ELAPSED/60)) min) ==="

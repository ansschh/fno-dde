#!/bin/bash
# Run adversarial + noise dense across 8 GPUs on selected sweeps.
# Fills F11 (robustness) figure data.
set -e
cd /workspace/dde-fno
mkdir -p logs/robustness

# Use the same priority sweeps as the rest of the pipeline.
ROOTS=(
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.7_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.9_runpod
  extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.99_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memory_aware_runpod
  extracted/pod_pulls_2026_05_03_final/H100main/outputs/memno_ffno_runpod
  extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/b5_causal_smooth_runpod
  extracted/pod_pulls_2026_05_03_final/244/outputs/film_ablation_runpod
  extracted/pod_pulls_2026_05_03_final/A40/outputs/film_ablation_runpod
)

# eval_adversarial_dense.py and eval_noise_dense.py crawl the roots.
# Each is single-process; we run them concurrently across N_GPUS=8 by
# splitting cells. Simpler approach: shard by family (5 fams) → 5 GPUs run
# adversarial, 3 GPUs run noise (or interleave).
#
# Actual approach: run them sequentially. Each script handles all cells via
# its internal loop with skip-if-exists. Pin to GPU 0 for sequential.

export CUDA_VISIBLE_DEVICES=0
echo "[robustness] starting adversarial dense $(date)"
python3 -u scripts/eval_adversarial_dense.py \
  --roots "${ROOTS[@]}" \
  --data_dir data_dde_pde \
  --regimes clean \
  2>&1 | tee logs/robustness/adversarial.log | tail -30

echo ""
echo "[robustness] starting noise dense $(date)"
python3 -u scripts/eval_noise_dense.py \
  --roots "${ROOTS[@]}" \
  --data_dir data_dde_pde \
  --regimes clean \
  2>&1 | tee logs/robustness/noise.log | tail -30

echo ""
echo "[robustness] done $(date)"
n_adv=$(find /workspace/dde-fno/extracted -name adversarial_dense.json 2>/dev/null | wc -l)
n_noise=$(find /workspace/dde-fno/extracted -name noise_dense.json 2>/dev/null | wc -l)
echo "[robustness] final counts: adversarial=$n_adv, noise=$n_noise"

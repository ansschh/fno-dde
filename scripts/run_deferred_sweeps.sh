#!/usr/bin/env bash
# Master launcher for deferred sweeps on H200 pod (8×144GB).
# Phases sequential (each waits for previous):
#   1. Causal LEMO sweep   — 45 cells, ~1.5h
#   2. σ-stability sweep   — 60 cells, ~2h
#   3. Lag-modes sweep     — 60 cells, ~2h
#   4. OOD eval (post-hoc) — ~30min
#
# Uses 48 workers (6 per H200 GPU, batch=8) to maximize throughput.
# Total ETA ~6h ≈ $24 at $3.99/GPU/hr × 8.
#
# Usage:
#   bash scripts/run_deferred_sweeps.sh phase_1   # Causal LEMO
#   bash scripts/run_deferred_sweeps.sh phase_2   # σ-stability
#   bash scripts/run_deferred_sweeps.sh phase_3   # Lag-modes
#   bash scripts/run_deferred_sweeps.sh phase_4   # OOD (post-hoc)
#   bash scripts/run_deferred_sweeps.sh all       # phases 1->2->3->4 sequentially

set -e

DIST_KERNEL="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
REGIMES="clean,lowres,noisy"
SEEDS="42,123,456"
N_WORKERS=24
N_GPUS=8
EPOCHS=200
BATCH=16

case "${1:-help}" in

  # -------------------------------------------------------------------------
  # Phase 1: Causal LEMO — replaces standard lemo_pc_nd with the T1' variant.
  # Same hyperparams as v2 sweep so direct comparison vs LEMO_PC is clean.
  # -------------------------------------------------------------------------
  phase_1)
    echo "=== Phase 1: Causal LEMO (T1') sweep ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL} \
      --models causal_lemo_pc_nd \
      --regimes ${REGIMES} --seeds ${SEEDS} \
      --epochs ${EPOCHS} --batch_size ${BATCH} \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} --residual_anchor \
      --data_dir data_dde_pde \
      --output_dir outputs/deferred_p1_causal_lemo
    ;;

  # -------------------------------------------------------------------------
  # Phase 2: σ-stability — train LEMO_PC at 4 σ values × 5 fams × 3 seeds.
  # Single regime (clean) only — σ behavior is regime-independent.
  # -------------------------------------------------------------------------
  phase_2)
    echo "=== Phase 2: σ-stability sweep ==="
    for sigma in 0.5 0.7 0.9 0.99; do
      echo "  -- sigma=$sigma --"
      python3 scripts/run_apebench_sweep.py \
        --datasets ${DIST_KERNEL} \
        --models lemo_pc_nd \
        --regimes clean --seeds ${SEEDS} \
        --epochs ${EPOCHS} --batch_size ${BATCH} \
        --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
        --noise_std 0.05 --downsample_factor 2 \
        --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \
        --residual_anchor --sigma ${sigma} \
        --data_dir data_dde_pde \
        --output_dir "outputs/deferred_p2_sigma_${sigma}"
    done
    ;;

  # -------------------------------------------------------------------------
  # Phase 3: Lag-modes (T4 quantization) sweep — vary spectral truncation.
  # 4 lag_modes values × 5 fams × 3 seeds × 1 regime (clean) = 60 cells.
  # Smaller lag_modes = more aggressive truncation = T4 grid quantization analog.
  # -------------------------------------------------------------------------
  phase_3)
    echo "=== Phase 3: Lag-modes (T4-style) sweep ==="
    for lm in 4 8 16 32; do
      echo "  -- lag_modes=$lm --"
      python3 scripts/run_apebench_sweep.py \
        --datasets ${DIST_KERNEL} \
        --models lemo_pc_nd \
        --regimes clean --seeds ${SEEDS} \
        --epochs ${EPOCHS} --batch_size ${BATCH} \
        --width 64 --n_layers 3 --lag_modes ${lm} --spatial_modes 12 \
        --noise_std 0.05 --downsample_factor 2 \
        --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \
        --residual_anchor \
        --data_dir data_dde_pde \
        --output_dir "outputs/deferred_p3_lagmodes_${lm}"
    done
    ;;

  # -------------------------------------------------------------------------
  # Phase 4: OOD eval (post-hoc, no retraining).
  # Needs gen_dde_pde_ood.py extension — TODO: separate script.
  # For now, this just generates OOD test sets with shifted kernel parameters.
  # -------------------------------------------------------------------------
  phase_4)
    echo "=== Phase 4: OOD eval (post-hoc) ==="
    echo "  TODO: extend gen_dde_pde_ood.py to support kernel families."
    echo "  Once extended, generate OOD test sets and run eval_lag_shift_ood.py"
    echo "  on outputs/deferred_p1_causal_lemo and outputs/deferred_p2_sigma_*."
    ;;

  all)
    bash "$0" phase_1
    bash "$0" phase_2
    bash "$0" phase_3
    bash "$0" phase_4
    ;;

  help|*)
    echo "Phases:"
    echo "  phase_1   Causal LEMO (45 cells, ~1.5h)"
    echo "  phase_2   sigma-stability (60 cells: 4 sigma x 5 fams x 3 seeds)"
    echo "  phase_3   lag-modes T4-style (60 cells: 4 lag_modes x 5 fams x 3 seeds)"
    echo "  phase_4   OOD eval (post-hoc, ~30min)"
    echo "  all       phase_1 -> phase_2 -> phase_3 -> phase_4"
    ;;
esac

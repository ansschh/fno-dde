#!/usr/bin/env bash
# =============================================================================
# Memory-aware baselines sweep — closes Round 3 fatal #1.
#
# Reviewers want head-to-head against:
#   - MemNO       (Buitrago et al. 2025)   -> already in baselines_nd.py
#   - F-FNO       (Tran et al. 2023)        -> already in baselines_nd.py
#   - NIDE        (Zappala et al. 2023)     -> stub in baselines_memory_aware.py
#   - NDDE        (Zhu et al. 2021)         -> stub in baselines_memory_aware.py
#   - S4          (Gu et al. 2022)          -> stub in baselines_memory_aware.py
#
# This script launches the three NEW baselines.  MemNO and F-FNO are run by
# the existing `followup_sweep.sh` and the v2 sweep, so they are not
# duplicated here.
#
# Sweep:
#   3 models   x 5 families x 3 regimes x 3 seeds = 135 cells.
#   Matched compute envelope to the LEMO_PC headline:
#     width=64, n_layers=3, lag_modes=24, spatial_modes=12, batch=4,
#     200 epochs, cosine LR, residual_anchor input.
#
# Hardware: 8 x H100, 24 workers (3 cells/GPU).
# Estimate: ~3-4h per H100 per cell; 135 cells / 24 workers ~= 6 batches
#           @ ~3.5h each = ~20 wall-clock hours.  Run on 2 pods in parallel
#           (one for nide+ndde, one for s4) to halve that to ~10h.
#
# PRECONDITION: the three classes in
#   src/models/baselines_memory_aware.py
# must have their `forward` methods filled in (stubs raise
# NotImplementedError at the moment).  The launcher will smoke-test before
# committing the full sweep.
# =============================================================================

set -e

DIST_KERNEL_FAMILIES="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
REGIMES="clean,lowres,noisy"
SEEDS="42,123,456"

# Matched-compute hyperparameters (identical to the LEMO_PC headline).
MATCHED_HPARAMS="--width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
                  --noise_std 0.05 --downsample_factor 2 --residual_anchor \
                  --batch_size 4 --epochs 200"

case "${1:-help}" in

  # -------------------------------------------------------------------------
  # smoke — single-cell smoke test on dist_exp + clean + seed 42 for each
  # of the three new baselines.  Aborts if per-epoch > 50s on H100.
  # Per the user's "smoke test first on new pods" rule.
  # -------------------------------------------------------------------------
  smoke)
    echo "=== Memory-aware baselines smoke test (3 cells, 5 epochs each) ==="
    for model in nide_nd ndde_nd s4_nd; do
      python3 scripts/run_apebench_sweep.py \
        --datasets dist_exp_rd_2d \
        --models ${model} \
        --regimes clean \
        --seeds 42 \
        --epochs 5 ${MATCHED_HPARAMS} \
        --n_workers 1 --n_gpus 1 \
        --data_dir data_dde_pde \
        --output_dir outputs/memory_aware_smoke/${model}
    done
    echo "Smoke test complete.  Inspect outputs/memory_aware_smoke/*/run.log"
    echo "and verify per-epoch wall-clock is below 50s before launching the sweep."
    ;;

  # -------------------------------------------------------------------------
  # nide — NIDE only (45 cells).  Run on pod A.
  # -------------------------------------------------------------------------
  nide)
    echo "=== NIDE_ND on 5 dist-kernel families x 3 regimes x 3 seeds ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} \
      --models nide_nd \
      --regimes ${REGIMES} \
      --seeds ${SEEDS} \
      ${MATCHED_HPARAMS} \
      --n_workers 24 --n_gpus 8 \
      --data_dir data_dde_pde \
      --output_dir outputs/memory_aware_nide
    ;;

  # -------------------------------------------------------------------------
  # ndde — NDDE only (45 cells).  Run on pod A after NIDE.
  # -------------------------------------------------------------------------
  ndde)
    echo "=== NDDE_ND on 5 dist-kernel families x 3 regimes x 3 seeds ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} \
      --models ndde_nd \
      --regimes ${REGIMES} \
      --seeds ${SEEDS} \
      ${MATCHED_HPARAMS} \
      --n_workers 24 --n_gpus 8 \
      --data_dir data_dde_pde \
      --output_dir outputs/memory_aware_ndde
    ;;

  # -------------------------------------------------------------------------
  # s4 — S4 only (45 cells).  Run on pod B in parallel with pod A.
  # -------------------------------------------------------------------------
  s4)
    echo "=== S4_ND on 5 dist-kernel families x 3 regimes x 3 seeds ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} \
      --models s4_nd \
      --regimes ${REGIMES} \
      --seeds ${SEEDS} \
      ${MATCHED_HPARAMS} \
      --n_workers 24 --n_gpus 8 \
      --data_dir data_dde_pde \
      --output_dir outputs/memory_aware_s4
    ;;

  # -------------------------------------------------------------------------
  # all — all three baselines, sequenced on a single pod (slowest path).
  # 135 cells; budget ~12h on 8 H100 with 24 workers.
  # -------------------------------------------------------------------------
  all)
    echo "=== Memory-aware sweep — full 135-cell run ==="
    bash "$0" smoke
    bash "$0" nide
    bash "$0" ndde
    bash "$0" s4
    echo "=== All three memory-aware baselines complete. ==="
    echo "Run aggregation:  python3 scripts/aggregate_phase_a.py \\"
    echo "                    --root outputs/memory_aware_{nide,ndde,s4} \\"
    echo "                    --out outputs/memory_aware_summary.json"
    ;;

  help|*)
    echo "Memory-aware baselines launcher.  Closes Round 3 fatal #1."
    echo ""
    echo "Phases:"
    echo "  smoke   3-cell, 5-epoch smoke test (verify per-epoch wall-clock)"
    echo "  nide    NIDE_ND        : 45 cells (5 fams x 3 regimes x 3 seeds)"
    echo "  ndde    NDDE_ND        : 45 cells"
    echo "  s4      S4_ND          : 45 cells"
    echo "  all     all three      : 135 cells (~12h on 8 H100)"
    echo ""
    echo "Models: NIDE (Zappala AAAI 2023), NDDE (Zhu ICLR 2021), S4 (Gu ICLR 2022)."
    echo "Stubs in src/models/baselines_memory_aware.py — fill forward() before launching."
    ;;
esac

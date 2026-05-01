#!/usr/bin/env bash
# =============================================================================
# FiLM-vs-equivariance ablation suite — launcher.
#
# Round-3 fix for REVIEW_PUNCH_LIST.md top risk #2:
# "No ablation that cleanly isolates lag-equivariance from FiLM and the
#  pointwise channel-mixing path; need FNO+FiLM and a non-equivariant
#  FiLM-conditioned lag mixer, and a corrected FiLM-free variant that
#  includes the B_ell term; ensure all models see the same history window."
#
# Variants  (4 total)
#   A1  fno_film_nd        : vanilla FNO + per-(out, lag_mode) FiLM head
#   A2  noneq_film_nd      : per-lag-position MLP (no shared lag weights) + FiLM
#   A3  lemo_bcorrect_nd   : LEMO with corrected B_ell, FiLM-free, GELU
#   A4  lemo_pc_nd         : full LEMO-PC (headline reference)
#
# Sweep matrix
#   4 variants × 5 dist-kernel families × 3 regimes × 3 seeds = 180 cells
#
# Hyperparameters — IDENTICAL to the headline 45-cell sweep
# (matched per `feedback_no_unfair_lemo_favors`):
#   width        = 64
#   n_layers     = 3
#   lag_modes    = 24
#   spatial_modes= 12
#   batch_size   = 4
#   epochs       = 200
#   optimizer    = Adam (in trainer)  lr = 1e-3, weight_decay = 1e-3
#   schedule     = CosineAnnealingLR (in trainer), eta_min = 1e-6
#   residual_anchor = on
#   sigma        = none (matches headline; sigma sweep is a separate experiment)
#   noise_std    = 0.05  (noisy regime)
#   downsample   = 2     (lowres regime)
#
# All four variants see the SAME residual-anchored history window
# (n_hist = 64 frames at dt=0.01 on a 64×64 grid), with the SAME
# per-sample params channel laid out at slot [0, 0, ..., -params_dim:].
#
# GPU usage  (per `feedback_max_gpu_usage`)
#   8 H100 GPUs × 3 cells/GPU = 24 parallel workers
#
# Estimated wall-clock
#   per-cell ~3.5 h (matches the headline run for LEMOPC-class models with
#   the same width/layers/modes; FNO+FiLM is slightly faster but we budget
#   the slowest variant for safety).
#   total = 180 cells / 24 workers × 3.5 h = 26.25 h wall-clock
#   This exceeds the 15 h GPU budget; see DEFERRED note at bottom.
# =============================================================================

set -e

DIST_KERNEL_FAMILIES="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
REGIMES="clean,lowres,noisy"
SEEDS="42,123,456"
VARIANTS="fno_film_nd,noneq_film_nd,lemo_bcorrect_nd,lemo_pc_nd"

DATA_DIR=${DATA_DIR:-data_dde_pde}
OUT_ROOT=${OUT_ROOT:-outputs/film_ablation}

case "${1:-help}" in

  smoke)
    # Smoke test (per `feedback_smoke_test_first`): 1 variant × 1 family ×
    # 1 regime × 1 seed × 5 epochs.  Aborts the full sweep if per-epoch
    # wall-clock exceeds 50 s.
    echo "=== smoke test: 5 epochs of fno_film_nd on dist_exp_rd_2d/clean ==="
    python3 -u scripts/train_apebench_smoke.py \
      --family dist_exp_rd_2d \
      --model fno_film_nd \
      --regime clean \
      --noise_std 0.05 --downsample_factor 2 \
      --epochs 5 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --seed 42 \
      --residual_anchor \
      --data_dir ${DATA_DIR} \
      --output_dir ${OUT_ROOT}_smoke
    ;;

  full)
    # Full 180-cell sweep.
    echo "=== FiLM ablation full sweep ==="
    echo "  variants : ${VARIANTS}"
    echo "  families : ${DIST_KERNEL_FAMILIES}"
    echo "  regimes  : ${REGIMES}"
    echo "  seeds    : ${SEEDS}"
    echo "  total    : 4 × 5 × 3 × 3 = 180 cells"
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} \
      --models   ${VARIANTS} \
      --regimes  ${REGIMES} \
      --seeds    ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir ${DATA_DIR} \
      --output_dir ${OUT_ROOT}
    ;;

  # -------------------------------------------------------------------------
  # 4 staged phases of 45 cells each, in case the 15h GPU budget forces
  # us to interleave with other revision experiments.  Each phase is
  # 45 cells / 24 workers × 3.5h ≈ 6.6h wall-clock.
  # -------------------------------------------------------------------------
  phase_a1)
    echo "=== A1: FNO + FiLM (45 cells) ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} --models fno_film_nd \
      --regimes ${REGIMES} --seeds ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir ${DATA_DIR} --output_dir ${OUT_ROOT}/a1_fno_film
    ;;

  phase_a2)
    echo "=== A2: non-eq FiLM lag mixer (45 cells) ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} --models noneq_film_nd \
      --regimes ${REGIMES} --seeds ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir ${DATA_DIR} --output_dir ${OUT_ROOT}/a2_noneq_film
    ;;

  phase_a3)
    echo "=== A3: LEMO + B_ell, FiLM-free (45 cells) ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} --models lemo_bcorrect_nd \
      --regimes ${REGIMES} --seeds ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir ${DATA_DIR} --output_dir ${OUT_ROOT}/a3_bcorrect
    ;;

  phase_a4)
    echo "=== A4: full LEMO-PC reference (45 cells) ==="
    # If the headline run already covers this exact (variant, family,
    # regime, seed, hyperparam) cell, skip — `run_apebench_sweep.py` will
    # detect existing test_results.json by default.
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} --models lemo_pc_nd \
      --regimes ${REGIMES} --seeds ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir ${DATA_DIR} --output_dir ${OUT_ROOT}/a4_full_lemopc
    ;;

  help|*)
    cat <<EOF
FiLM-vs-equivariance ablation suite.

Usage:
  $0 smoke       # 1 cell × 5 epochs sanity check (must run first)
  $0 full        # all 180 cells in a single sweep (~26h on 8×H100)
  $0 phase_a1    # A1 only (45 cells, ~6.6h)
  $0 phase_a2    # A2 only (45 cells, ~6.6h)
  $0 phase_a3    # A3 only (45 cells, ~6.6h)
  $0 phase_a4    # A4 only (45 cells, ~6.6h)  [reference; may dedup]

Environment:
  DATA_DIR    (default: data_dde_pde)
  OUT_ROOT    (default: outputs/film_ablation)

Compute estimate (8 × H100, 3 cells/GPU, 24 workers):
  per-cell wall-clock ~ 3.5 h    (matches headline LEMO-PC at width=64)
  total              = 180 cells / 24 workers × 3.5 h ≈ 26.25 h
  staged             = 4 × ~6.6 h phases (each fits inside one 15h budget,
                       with ~7-8h headroom for OOD/E_orbit follow-ups)

Notes:
  - Models fno_film_nd / noneq_film_nd / lemo_bcorrect_nd are defined in
    scripts/film_ablation_design.py and dispatched through src/train/build_model.py
    via the corresponding model_class string. To wire them in, add the
    three entries (one elif branch each) to build_model() that import and
    call create_fno_film_nd / create_noneq_film_nd / create_lemo_bcorrect_nd
    from scripts.film_ablation_design.
  - All four variants ingest the SAME history window — the residual-anchor
    flag and data_dir are identical.  This is the apples-to-apples
    comparison the reviewers asked for.
  - Out of GPU budget for the full 180-cell run in one shot.  Recommended
    plan: phase_a1 + phase_a2 in one 15h budget; phase_a3 + post-hoc OOD
    follow-up in the next.  A4 is reused from headline ckpts where possible.
EOF
    ;;
esac

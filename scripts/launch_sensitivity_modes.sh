#!/usr/bin/env bash
# =============================================================================
# Sensitivity sweep S2: number of spectral spatial modes.
#
# Reviewer fixable issue [3]: "Run modest sensitivity sweeps (lag grid size,
# number of modes, FiLM rank, history length; β in the weighted norm)."
#
# This launcher covers the *spatial-modes* axis. The lag-modes axis is covered
# by `run_deferred_sweeps.sh phase_3`; the lag-grid / history-length axis is
# covered by Pod 4's Phase B; the FiLM-rank axis is covered by
# `launch_sensitivity_film_rank.sh`; the β-rate axis is post-hoc on σ-sweep
# checkpoints.
#
# Sweep grid:
#   spatial_modes ∈ {4, 8, 12, 16, 24}  (24 clipped to 16 = Nyquist on 64x64)
#   datasets       = 5 distributed-kernel families
#   models         = lemo_pc_nd, fno_nd  (FNO so we read off whether spectral
#                                          truncation interacts with the
#                                          baseline differently)
#   regimes        = clean
#   seeds          = 42, 123, 456
#   total cells    = 5 modes × 5 fams × 2 models × 3 seeds = 150
#
# Compute:
#   ~5 min per cell × 150 cells / 24 workers ≈ 30 min on 8×H100 pod.
#
# Usage:
#   bash scripts/launch_sensitivity_modes.sh             # full sweep
#   bash scripts/launch_sensitivity_modes.sh --smoke     # one cell, 5 epochs
#
# Output:
#   outputs/sensitivity_modes/<spatial_modes>/raw/<family>/<regime>/<model>/s<seed>/
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DIST_KERNEL="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
SEEDS="42,123,456"
MODELS="lemo_pc_nd,fno_nd"
N_WORKERS=24
N_GPUS=8
EPOCHS=200
BATCH=8

# headline defaults (all axes except spatial_modes held fixed)
WIDTH=64
N_LAYERS=3
LAG_MODES=24

# ---- arg parsing ------------------------------------------------------------
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    -h|--help)
      sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[launch_sensitivity_modes] unknown arg: $arg"; exit 2 ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  echo "=== Smoke: one cell of S2 (LEMO_PC, dist_exp, clean, seed 42, sm=12, 5 epochs) ==="
  python3 -u scripts/train_apebench_smoke.py \
    --data_dir data_dde_pde \
    --family dist_exp_rd_2d \
    --model lemo_pc_nd \
    --regime clean \
    --epochs 5 --batch_size ${BATCH} \
    --width ${WIDTH} --n_layers ${N_LAYERS} \
    --lag_modes ${LAG_MODES} --spatial_modes 12 \
    --seed 42 --residual_anchor \
    --output_dir outputs/sensitivity_modes/smoke
  echo "=== smoke done.  Per-epoch wall should be <50s; if not, abort full launch. ==="
  exit 0
fi

# ---- full sweep -------------------------------------------------------------
echo "=== Sensitivity-S2: spatial-modes sweep (5 modes × 5 fams × 2 models × 3 seeds = 150 cells) ==="

for sm in 4 8 12 16 24; do
  echo ""
  echo "  -- spatial_modes=${sm} --"
  python3 scripts/run_apebench_sweep.py \
    --datasets ${DIST_KERNEL} \
    --models ${MODELS} \
    --regimes clean --seeds ${SEEDS} \
    --epochs ${EPOCHS} --batch_size ${BATCH} \
    --width ${WIDTH} --n_layers ${N_LAYERS} \
    --lag_modes ${LAG_MODES} --spatial_modes ${sm} \
    --noise_std 0.05 --downsample_factor 2 \
    --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \
    --residual_anchor \
    --data_dir data_dde_pde \
    --output_dir "outputs/sensitivity_modes/sm_${sm}"
done

echo ""
echo "=== S2 done.  Aggregate via:"
echo "    python3 scripts/aggregate_sensitivity.py --root outputs/sensitivity_modes --axis spatial_modes"

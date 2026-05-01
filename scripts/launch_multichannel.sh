#!/usr/bin/env bash
# launch_multichannel.sh
# -----------------------------------------------------------------------------
# Multi-channel reaction-diffusion sweep (Round-3 fixable issue [4]).
#
# The headline 45-cell benchmark is single-channel (u in R per grid cell). This
# script runs a controlled Gray-Scott (gray_scott_3d) sweep with TWO coupled
# channels (activator u, inhibitor v) to verify that the FiLM-modulated
# parametric-conditional kernel scales to coupled fields, and to bound the
# scope of the lag-equivariance prior under Markovian-but-multi-channel data.
#
# Sweep design
# ------------
#   1 family x 3 regimes x 3 seeds x 3 models = 27 cells
#     family : gray_scott_3d (data_dde_pde/gray_scott_3d if present, else
#              regenerate via scripts/gen_dde_pde.py --family gray_scott_3d)
#     regimes: clean, lowres, noisy
#     seeds  : 42, 123, 456
#     models : lemo_pc_nd, fno_3d, unet_3d
#
# Expected outcome (based on APEBench negative-result analysis in
# sections/experimental_design_and_evaluation_plan.tex, Sec. APEBench negative):
#   - gray_scott_3d is autoregressively Markovian on the time axis, so we do
#     NOT expect a lag-equivariance benefit; success criterion is that
#     LEMO_PC_ND remains within ~5% of FNO and trains stably on multi-channel
#     input/output without OOM, FiLM divergence, or kernel collapse.
#
# Compute budget
# --------------
#   24 workers / 8 H100 GPUs (3 cells per GPU); ~2.5h ETA.
#   Single-script run; no need to chain with the deferred-sweeps launcher.
#
# Usage
# -----
#   bash scripts/launch_multichannel.sh smoke   # 1 cell, 5 epochs, sanity check
#   bash scripts/launch_multichannel.sh full    # 27 cells, 200 epochs (default)
#   bash scripts/launch_multichannel.sh check   # dry-run, prints the command
# -----------------------------------------------------------------------------

set -e

FAMILY="gray_scott_3d"
DATA_DIR="data_dde_pde"
REGIMES="clean,lowres,noisy"
SEEDS="42,123,456"
MODELS="lemo_pc_nd,fno_3d,unet_3d"

EPOCHS=200
BATCH=4
WIDTH=64
N_LAYERS=3
LAG_MODES=24
SPATIAL_MODES=12
NOISE_STD=0.05
DOWNSAMPLE_FACTOR=2

N_WORKERS=24
N_GPUS=8

OUT_BASE="outputs/multichannel_v1"

# ---- Data sanity check ------------------------------------------------------
# All baselines must see the same multi-channel tensor shape; warn explicitly
# if the data tarball was not regenerated to include both u and v channels.
if [[ ! -d "${DATA_DIR}/${FAMILY}/train" ]]; then
  echo "ERROR: ${DATA_DIR}/${FAMILY}/train not found."
  echo "Regenerate with:"
  echo "  python3 scripts/gen_dde_pde.py --family ${FAMILY} --n_train 1000 --n_val 200 --n_test 200"
  echo "and re-tar AFTER data gen (per data_tarball_includes_kernels rule)."
  exit 1
fi

# Verify channel count == 2 from a sample fingerprint (must match Gray-Scott
# spec). This catches the silent failure of using a single-channel npz that
# happens to live under gray_scott_3d/.
NCHAN=$(python3 -c "
import numpy as np, glob, os
fs = sorted(glob.glob('${DATA_DIR}/${FAMILY}/train/*.npz'))
if not fs:
    print(0); raise SystemExit
d = np.load(fs[0])
arr = d[d.files[0]]
# convention: (T, C, H, W, D) for 3D fields
print(arr.shape[1] if arr.ndim >= 5 else 1)
" 2>/dev/null || echo 0)

if [[ "${NCHAN}" != "2" ]]; then
  echo "ERROR: expected num_channels=2 for ${FAMILY}, got ${NCHAN}."
  echo "Regenerate the data tarball with both u and v channels before running this sweep."
  exit 2
fi

# ---- Sweep cases ------------------------------------------------------------
case "${1:-help}" in

  smoke)
    echo "=== Multi-channel SMOKE (1 cell, 5 epochs) ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets "${FAMILY}" \
      --models lemo_pc_nd \
      --regimes clean --seeds 42 \
      --epochs 5 --batch_size ${BATCH} \
      --width ${WIDTH} --n_layers ${N_LAYERS} \
      --lag_modes ${LAG_MODES} --spatial_modes ${SPATIAL_MODES} \
      --noise_std ${NOISE_STD} --downsample_factor ${DOWNSAMPLE_FACTOR} \
      --n_workers 1 --n_gpus 1 \
      --residual_anchor \
      --data_dir ${DATA_DIR} \
      --output_dir "${OUT_BASE}/_smoke"
    ;;

  full)
    echo "=== Multi-channel FULL sweep (27 cells, ~2.5h ETA) ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets "${FAMILY}" \
      --models "${MODELS}" \
      --regimes "${REGIMES}" --seeds "${SEEDS}" \
      --epochs ${EPOCHS} --batch_size ${BATCH} \
      --width ${WIDTH} --n_layers ${N_LAYERS} \
      --lag_modes ${LAG_MODES} --spatial_modes ${SPATIAL_MODES} \
      --noise_std ${NOISE_STD} --downsample_factor ${DOWNSAMPLE_FACTOR} \
      --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \
      --residual_anchor \
      --multichannel \
      --data_dir ${DATA_DIR} \
      --output_dir "${OUT_BASE}/raw"
    echo "Done. Outputs under ${OUT_BASE}/raw."
    echo "Aggregate with:"
    echo "  python3 scripts/aggregate_phase_a.py --input_dir ${OUT_BASE}/raw --output ${OUT_BASE}/agg.json"
    ;;

  check)
    echo "DRY RUN: would launch the following command --"
    echo "python3 scripts/run_apebench_sweep.py \\"
    echo "  --datasets ${FAMILY} \\"
    echo "  --models ${MODELS} \\"
    echo "  --regimes ${REGIMES} --seeds ${SEEDS} \\"
    echo "  --epochs ${EPOCHS} --batch_size ${BATCH} \\"
    echo "  --width ${WIDTH} --n_layers ${N_LAYERS} \\"
    echo "  --lag_modes ${LAG_MODES} --spatial_modes ${SPATIAL_MODES} \\"
    echo "  --noise_std ${NOISE_STD} --downsample_factor ${DOWNSAMPLE_FACTOR} \\"
    echo "  --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \\"
    echo "  --residual_anchor --multichannel \\"
    echo "  --data_dir ${DATA_DIR} \\"
    echo "  --output_dir ${OUT_BASE}/raw"
    ;;

  help|*)
    echo "Multi-channel (Gray-Scott, 2-channel) sweep launcher."
    echo "Usage:"
    echo "  bash scripts/launch_multichannel.sh smoke   # 1 cell x 5 epochs"
    echo "  bash scripts/launch_multichannel.sh full    # 27 cells (1x3x3x3) x 200 epochs"
    echo "  bash scripts/launch_multichannel.sh check   # dry run"
    echo ""
    echo "Total cells: 1 family x 3 regimes x 3 seeds x 3 models = 27."
    echo "Models: lemo_pc_nd, fno_3d, unet_3d (matched at width=${WIDTH}, depth=${N_LAYERS})."
    ;;
esac

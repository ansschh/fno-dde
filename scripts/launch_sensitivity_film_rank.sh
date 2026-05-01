#!/usr/bin/env bash
# =============================================================================
# Sensitivity sweep S3: FiLM rank (film_hidden).
#
# Reviewer fixable issue [3]: sweep the FiLM modulator's hidden width
# (currently 64) ∈ {16, 32, 64, 128} to verify the headline result is not an
# artifact of a high-rank FiLM head.
#
# The FiLM head is a 2-layer MLP:
#   params_dim → film_hidden → 2 · out_channels · lag_modes
# in `src/models/lemo_pc_nd.py::FiLMLagSpectralND.film_net`.  Small
# film_hidden imposes a low-rank bottleneck on the per-sample conditioning.
#
# The trainer (`scripts/train_apebench_smoke.py`) does not expose
# `--film_hidden` as a CLI argument out of the box; the sweep launcher
# applies a minimal, idempotent patch that:
#   (1) adds `--film_hidden` to the argparse,
#   (2) threads it into `config["model"]["film_hidden"]`.
# The patch leaves the model code untouched and the headline default is
# preserved (film_hidden defaults to 64 if the flag is not passed), so this
# patch does NOT alter any other sweep's behaviour.
#
# Sweep grid:
#   film_hidden ∈ {16, 32, 64, 128}
#   datasets    = 5 distributed-kernel families
#   model       = lemo_pc_nd
#   regimes     = clean
#   seeds       = 42, 123, 456
#   total cells = 4 × 5 × 1 × 3 = 60
#
# Compute:
#   ~5 min per cell × 60 cells / 24 workers ≈ 13 min on 8×H100 pod.
#
# Usage:
#   bash scripts/launch_sensitivity_film_rank.sh             # full sweep
#   bash scripts/launch_sensitivity_film_rank.sh --smoke     # one cell, 5 epochs
#   bash scripts/launch_sensitivity_film_rank.sh --patch-only # apply the patch and exit
#
# Output:
#   outputs/sensitivity_film_rank/fh_<film_hidden>/raw/<family>/<regime>/<model>/s<seed>/
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DIST_KERNEL="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
SEEDS="42,123,456"
N_WORKERS=24
N_GPUS=8
EPOCHS=200
BATCH=8

# headline defaults (all axes except film_hidden held fixed)
WIDTH=64
N_LAYERS=3
LAG_MODES=24
SPATIAL_MODES=12

TRAINER="scripts/train_apebench_smoke.py"

# ---- idempotent patch: add --film_hidden CLI to the trainer -----------------
apply_film_hidden_patch() {
  if grep -q -- '--film_hidden' "$TRAINER"; then
    echo "[patch] $TRAINER already has --film_hidden; skipping."
    return 0
  fi
  echo "[patch] adding --film_hidden CLI to $TRAINER"
  python3 - <<PYEOF
import io, re, sys
from pathlib import Path

p = Path("${TRAINER}")
src = p.read_text()

# 1. Add --film_hidden right after --spatial_modes argparse line.
needle_arg = 'ap.add_argument("--spatial_modes", type=int, default=8)'
ins_arg = 'ap.add_argument("--spatial_modes", type=int, default=8)\n    ap.add_argument("--film_hidden", type=int, default=64,\n                    help="FiLM MLP hidden width (LEMO-PC only). Default 64 = headline.")'
if needle_arg not in src:
    print("[patch] FATAL: argparse anchor not found", file=sys.stderr)
    sys.exit(2)
src = src.replace(needle_arg, ins_arg, 1)

# 2. Thread args.film_hidden into config["model"].
needle_cfg = '"kernel_hidden":  64,'
ins_cfg = '"kernel_hidden":  64,\n            "film_hidden":    args.film_hidden,'
if needle_cfg not in src:
    print("[patch] FATAL: config anchor not found", file=sys.stderr)
    sys.exit(2)
src = src.replace(needle_cfg, ins_cfg, 1)

p.write_text(src)
print("[patch] patched OK.")
PYEOF
}

apply_film_hidden_patch_to_sweep() {
  # Sweep launcher passes through unrecognised flags to the trainer via
  # the `cmd` list construction.  We need to teach run_apebench_sweep.py
  # to pass `--film_hidden` through. Idempotent.
  local launcher="scripts/run_apebench_sweep.py"
  if grep -q -- 'args.film_hidden' "$launcher"; then
    echo "[patch] $launcher already passes --film_hidden; skipping."
    return 0
  fi
  echo "[patch] adding film_hidden pass-through to $launcher"
  python3 - <<PYEOF
from pathlib import Path
import sys

p = Path("scripts/run_apebench_sweep.py")
src = p.read_text()

# 1. argparse: add --film_hidden after --spatial_modes
needle = 'ap.add_argument("--spatial_modes", type=int, default=12)'
ins = 'ap.add_argument("--spatial_modes", type=int, default=12)\n    ap.add_argument("--film_hidden", type=int, default=64)'
if needle not in src:
    print("[patch] FATAL: sweep argparse anchor not found", file=sys.stderr)
    sys.exit(2)
src = src.replace(needle, ins, 1)

# 2. cmd construction: add --film_hidden right after --spatial_modes
needle2 = '"--spatial_modes", str(args.spatial_modes),'
ins2 = '"--spatial_modes", str(args.spatial_modes),\n                        "--film_hidden", str(args.film_hidden),'
if needle2 not in src:
    print("[patch] FATAL: sweep cmd anchor not found", file=sys.stderr)
    sys.exit(2)
src = src.replace(needle2, ins2, 1)

p.write_text(src)
print("[patch] sweep launcher patched OK.")
PYEOF
}

# ---- arg parsing ------------------------------------------------------------
SMOKE=0
PATCH_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --smoke)      SMOKE=1 ;;
    --patch-only) PATCH_ONLY=1 ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[launch_sensitivity_film_rank] unknown arg: $arg"; exit 2 ;;
  esac
done

apply_film_hidden_patch
apply_film_hidden_patch_to_sweep

if [[ "$PATCH_ONLY" -eq 1 ]]; then
  echo "=== --patch-only: trainer + sweep launcher patched.  Exiting. ==="
  exit 0
fi

if [[ "$SMOKE" -eq 1 ]]; then
  echo "=== Smoke: one cell of S3 (LEMO_PC, dist_exp, clean, seed 42, fh=32, 5 epochs) ==="
  python3 -u scripts/train_apebench_smoke.py \
    --data_dir data_dde_pde \
    --family dist_exp_rd_2d \
    --model lemo_pc_nd \
    --regime clean \
    --epochs 5 --batch_size ${BATCH} \
    --width ${WIDTH} --n_layers ${N_LAYERS} \
    --lag_modes ${LAG_MODES} --spatial_modes ${SPATIAL_MODES} \
    --film_hidden 32 \
    --seed 42 --residual_anchor \
    --output_dir outputs/sensitivity_film_rank/smoke
  echo "=== smoke done. Per-epoch wall should be <50s; if not, abort full launch. ==="
  exit 0
fi

# ---- full sweep -------------------------------------------------------------
echo "=== Sensitivity-S3: FiLM rank sweep (4 fh × 5 fams × 1 model × 3 seeds = 60 cells) ==="

for fh in 16 32 64 128; do
  echo ""
  echo "  -- film_hidden=${fh} --"
  python3 scripts/run_apebench_sweep.py \
    --datasets ${DIST_KERNEL} \
    --models lemo_pc_nd \
    --regimes clean --seeds ${SEEDS} \
    --epochs ${EPOCHS} --batch_size ${BATCH} \
    --width ${WIDTH} --n_layers ${N_LAYERS} \
    --lag_modes ${LAG_MODES} --spatial_modes ${SPATIAL_MODES} \
    --film_hidden ${fh} \
    --noise_std 0.05 --downsample_factor 2 \
    --n_workers ${N_WORKERS} --n_gpus ${N_GPUS} \
    --residual_anchor \
    --data_dir data_dde_pde \
    --output_dir "outputs/sensitivity_film_rank/fh_${fh}"
done

echo ""
echo "=== S3 done.  Aggregate via:"
echo "    python3 scripts/aggregate_sensitivity.py --root outputs/sensitivity_film_rank --axis film_hidden"

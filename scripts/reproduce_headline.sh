#!/usr/bin/env bash
# =============================================================================
# Reproduce the LEMO_PC headline result (single-GPU, ~2h on one H100).
#
# Headline claim:
#   "LEMO_PC reduces test relL2 by 69-80% over FNO/MarkovFNO/WindFNO on 5
#    distributed-kernel reaction-diffusion benchmarks (n=45 paired cells,
#    p<10^-4 Holm-corrected)."
#
# This script reproduces ONE cell of that claim end-to-end:
#   - Family   : dist_exp_rd_2d   (exponential-kernel reaction-diffusion 2D)
#   - Regime   : clean
#   - Seed     : 42
#   - Headline : LEMO_PC test relL2 ~ 0.012  vs  FNO ~ 0.07  (~80% reduction)
#
# Modes:
#   bash scripts/reproduce_headline.sh --gen-only   # data generation only
#   bash scripts/reproduce_headline.sh --train-only # skip data gen
#   bash scripts/reproduce_headline.sh              # full run (~2h)
#   bash scripts/reproduce_headline.sh --verify     # only re-verify last run
#
# Hardware:
#   - 1 GPU (H100 recommended; A100/RTX-3090 also OK, slower)
#   - ~12 GB GPU memory (batch 4, width 64)
#   - ~30 GB CPU RAM during data generation
#   - ~6 GB disk for data + checkpoints
#
# Total wall-clock budget on 1xH100:
#   - Data generation : ~25 min  (1000 + 200 + 200 trajectories, 64x64 grid)
#   - LEMO_PC train   : ~50 min  (200 epochs, 200 steps/epoch)
#   - FNO baseline    : ~40 min  (smaller model, faster)
#   - Verification    : <1 min
#   ----------------------------------------------------
#   - TOTAL           : ~2h
# =============================================================================

set -euo pipefail

# ---- repo root --------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- fixed configuration (matches headline cell) -----------------------------
FAMILY="dist_exp_rd_2d"
REGIME="clean"
SEED=42

# Data params (matches data_dde_pde manifest used in main paper sweep)
N_TRAIN=1000
N_VAL=200
N_TEST=200
N_GRID=64           # 64x64 spatial grid
N_HIST=16           # history window (lag axis)
N_OUT=16            # rollout horizon
DT=0.05
T_TOTAL=16.0

# Training params (matches v2 sweep / headline config)
EPOCHS=200
BATCH=4
LR=1e-3
WIDTH=64            # headline width
LAG_MODES=24        # headline lag-modes
SPATIAL_MODES=12
N_LAYERS=3

# Output paths
DATA_DIR="${REPO}/data_dde_pde"
OUT_ROOT="${REPO}/outputs/reproduce_headline"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Expected results (median of 3 seeds in main paper, single-seed will be close)
EXPECTED_LEMO_PC=0.012
EXPECTED_FNO=0.07
EXPECTED_IMPROVEMENT_PCT=80

# ---- arg parsing ------------------------------------------------------------
GEN_ONLY=0
TRAIN_ONLY=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --gen-only)    GEN_ONLY=1 ;;
    --train-only)  TRAIN_ONLY=1 ;;
    --verify)      VERIFY_ONLY=1 ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[reproduce_headline] unknown arg: $arg"; exit 2 ;;
  esac
done

# ---- helpers ----------------------------------------------------------------
log()  { echo "[reproduce_headline $(date +%H:%M:%S)] $*"; }
fail() { echo "[reproduce_headline ERROR] $*" >&2; exit 1; }

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "WARNING: nvidia-smi not found — will run on CPU (very slow)."
    return
  fi
  local n
  n=$(nvidia-smi -L | wc -l)
  log "GPUs detected: $n"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
}

# =============================================================================
# 1. Data generation
# =============================================================================
gen_data() {
  log "=== Stage 1/3 : data generation (${FAMILY}) ==="
  if [[ -f "${DATA_DIR}/${FAMILY}/test/shard_000.npz" \
        && -f "${DATA_DIR}/${FAMILY}/train/shard_000.npz" \
        && -f "${DATA_DIR}/${FAMILY}/val/shard_000.npz" ]]; then
    log "data already exists at ${DATA_DIR}/${FAMILY} — skipping generation."
    log "(delete that directory to force regeneration)"
    return
  fi
  log "generating ${N_TRAIN} train + ${N_VAL} val + ${N_TEST} test trajectories"
  log "grid ${N_GRID}x${N_GRID}, n_hist=${N_HIST}, n_out=${N_OUT}, dt=${DT}, T=${T_TOTAL}"
  python3 scripts/gen_dde_pde_data.py \
    --family "$FAMILY" \
    --num_train "$N_TRAIN" --num_val "$N_VAL" --num_test "$N_TEST" \
    --num_points "$N_GRID" \
    --T_total "$T_TOTAL" --dt "$DT" \
    --n_hist "$N_HIST" --n_out "$N_OUT" \
    --out_dir "$DATA_DIR" \
    2>&1 | tee "${LOG_DIR}/gen_data.log"
  log "data generation done."
}

# =============================================================================
# 2. Training
# =============================================================================
train_one() {
  local model="$1"
  log "=== training ${model} on ${FAMILY}/${REGIME}/s${SEED} ==="
  local cell_dir="${OUT_ROOT}/${FAMILY}/${REGIME}/${model}/s${SEED}"
  if [[ -f "${cell_dir}/test_results.json" ]]; then
    log "  ${model} already trained at ${cell_dir} — skipping."
    log "  (delete that directory to force retraining)"
    return
  fi
  python3 -u scripts/train_apebench_smoke.py \
    --data_dir "$DATA_DIR" \
    --family "$FAMILY" \
    --model "$model" \
    --regime "$REGIME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --width "$WIDTH" \
    --n_layers "$N_LAYERS" \
    --lag_modes "$LAG_MODES" \
    --spatial_modes "$SPATIAL_MODES" \
    --seed "$SEED" \
    --residual_anchor \
    --output_dir "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/train_${model}.log"
}

# =============================================================================
# 3. Verification — compare to expected numbers
# =============================================================================
verify() {
  log "=== Stage 3/3 : verification ==="
  python3 - <<PYEOF
import json, sys
from pathlib import Path

OUT = Path("${OUT_ROOT}")
FAM = "${FAMILY}"
REG = "${REGIME}"
SEED = ${SEED}
EXP_LEMO = ${EXPECTED_LEMO_PC}
EXP_FNO  = ${EXPECTED_FNO}
EXP_PCT  = ${EXPECTED_IMPROVEMENT_PCT}

results = {}
for model in ("lemo_pc_nd", "fno_nd"):
    p = OUT / FAM / REG / model / f"s{SEED}" / "test_results.json"
    if not p.exists():
        print(f"  MISSING: {p}")
        sys.exit(1)
    results[model] = json.load(open(p))["test_rel_l2_mean"]

lemo = results["lemo_pc_nd"]
fno  = results["fno_nd"]
improvement = 100.0 * (fno - lemo) / fno

print(f"  LEMO_PC test_rel_l2_mean = {lemo:.4f}    (expected ~{EXP_LEMO:.3f})")
print(f"  FNO      test_rel_l2_mean = {fno:.4f}    (expected ~{EXP_FNO:.3f})")
print(f"  Improvement  : {improvement:.1f}%       (expected ~{EXP_PCT}%)")

# Tolerance: +/- 30% of expected absolute value (single-seed run, paper is 3-seed median)
def near(actual, expected, rtol=0.30):
    return abs(actual - expected) / expected < rtol

ok_lemo = near(lemo, EXP_LEMO, rtol=0.40)
ok_fno  = near(fno,  EXP_FNO,  rtol=0.40)
ok_pct  = improvement >= 0.5 * EXP_PCT     # at least half the headline improvement

print()
if ok_lemo and ok_fno and ok_pct:
    print("  PASS  -- headline result reproduced within tolerance.")
    sys.exit(0)
else:
    print("  FAIL  -- numbers diverge from headline:")
    if not ok_lemo: print(f"    LEMO_PC off by >40% from expected {EXP_LEMO}")
    if not ok_fno:  print(f"    FNO     off by >40% from expected {EXP_FNO}")
    if not ok_pct:  print(f"    Improvement {improvement:.1f}% < {0.5*EXP_PCT:.0f}% (half headline)")
    sys.exit(2)
PYEOF
}

# =============================================================================
# Main
# =============================================================================
log "===================================================================="
log "DDE-FNO headline-result reproduction"
log "  family : ${FAMILY}/${REGIME} seed=${SEED}"
log "  expected: LEMO_PC=${EXPECTED_LEMO_PC}  FNO=${EXPECTED_FNO}  Δ≈${EXPECTED_IMPROVEMENT_PCT}%"
log "  output : ${OUT_ROOT}"
log "===================================================================="
check_gpu

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify
  exit $?
fi

if [[ "$TRAIN_ONLY" -eq 0 ]]; then
  gen_data
fi

if [[ "$GEN_ONLY" -eq 1 ]]; then
  log "data-gen complete, exiting (--gen-only)."
  exit 0
fi

log "=== Stage 2/3 : training (LEMO_PC, then FNO baseline) ==="
train_one lemo_pc_nd
train_one fno_nd

verify
log "done."

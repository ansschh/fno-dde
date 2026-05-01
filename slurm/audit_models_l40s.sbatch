#!/bin/bash
#SBATCH --job-name=lemo-audit-l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_audit.out
#SBATCH --error=slurm_logs/%j_audit.err

# =============================================================================
# PRE-DEPLOYMENT AUDIT
#
# Smoke-tests every model class × each regime + every sigma value before
# the full 386-cell sweep is allowed to consume 2+ days of compute.
#
# Catches:
#   - linalg divergence (SVD, eig, qr, cholesky) on ill-conditioned init
#   - NaN/Inf in loss
#   - CUDA OOM on H200
#   - Build/dispatch errors (model_class not in build_model)
#   - Data-path missing (dist_exp_rd_2d shards, orbit_ood/m4)
#   - Distribution mismatch on residual_anchor train/eval cycle
#   - Per-cell timeout (>10min suggests a hang)
#
# Outputs:
#   outputs/audit/results.txt       PASS/FAIL per cell with reason
#   outputs/audit/logs/<label>.log  Per-cell stdout/stderr
#   slurm_logs/<jobid>_audit.{out,err}  This script's main log
#
# Wall-clock estimate on H200: 36 cells × 1-3 min ≈ 30-90 min total.
#
# Usage on Caltech:
#   cd ~/dde-fno && git pull origin main
#   sbatch slurm/audit_models.sbatch
#   # wait ~1h, then check the summary at the END of the .out file.
#   # If all PASS, the full 386-cell sweep is safe to leave running.
# =============================================================================

set -uo pipefail

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

mkdir -p outputs/audit/logs
RESULTS=outputs/audit/results.txt
> $RESULTS  # truncate

# -----------------------------------------------------------------------------
# Helper: run one cell, capture pass/fail with reason.
# -----------------------------------------------------------------------------
run_one() {
  local label=$1; shift
  local log=outputs/audit/logs/$(echo "$label" | tr '/=' '__').log
  echo ""
  echo "[$(date)] AUDIT START: $label"
  echo "  cmd: $*"

  # 10-min hard timeout per cell. set +e because we want to record exit codes.
  set +e
  timeout 600 "$@" > "$log" 2>&1
  local rc=$?
  set -e

  if [ $rc -eq 0 ]; then
    # Even if exit was clean, check for NaN/Inf in the loss output.
    if grep -qiE '(train_loss=nan|val_relL2=nan|loss=inf)' "$log"; then
      echo "FAIL $label  REASON=NaN_or_Inf_in_loss" | tee -a $RESULTS
    elif grep -qE 'final test relL2' "$log" || grep -qE 'val_relL2=' "$log"; then
      echo "PASS $label" | tee -a $RESULTS
    else
      echo "FAIL $label  REASON=no_progress_signal_in_log" | tee -a $RESULTS
    fi
  elif [ $rc -eq 124 ]; then
    echo "FAIL $label  REASON=TIMEOUT_10min" | tee -a $RESULTS
  else
    # Grab last meaningful error line.
    local err
    err=$(grep -E 'Error|Traceback|RuntimeError|_LinAlgError|OutOfMemory' "$log" | tail -1 | head -c 200)
    if [ -z "$err" ]; then
      err=$(tail -3 "$log" | tr '\n' ' ' | head -c 200)
    fi
    echo "FAIL $label  REASON=exit_${rc}  TAIL=${err}" | tee -a $RESULTS
  fi
}

# -----------------------------------------------------------------------------
# Common args.
# -----------------------------------------------------------------------------
BASE="--family dist_exp_rd_2d \
  --noise_std 0.05 --downsample_factor 2 \
  --epochs 5 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed 42 \
  --data_dir data_dde_pde \
  --residual_anchor"

echo "================================="
echo "AUDIT START: $(date)"
echo "================================="

# -----------------------------------------------------------------------------
# Phase 1: every model class × every regime (5 epochs each).
# 10 models × 3 regimes = 30 cells.
# -----------------------------------------------------------------------------
for MODEL in lemo_pc_nd lemo_bcorrect_nd fno_film_nd noneq_film_nd \
             per_lag_mlp_nd nide_nd ndde_nd s4_nd \
             memno_nd ffno_nd; do
  for REGIME in clean lowres noisy; do
    OUT=outputs/audit/${MODEL}_${REGIME}_s42
    run_one "${MODEL}/${REGIME}" \
      python3 -u scripts/train_apebench_smoke.py \
        --model $MODEL --regime $REGIME --output_dir $OUT $BASE
  done
done

# -----------------------------------------------------------------------------
# Phase 2: σ-projection stress test (lemo_pc_nd + every sigma value, 10 epochs
# to catch slow-developing instabilities).
# 4 cells.
# -----------------------------------------------------------------------------
for SIGMA in 0.5 0.7 0.9 0.99; do
  OUT=outputs/audit/lemo_pc_nd_clean_sigma${SIGMA}_s42
  run_one "lemo_pc_nd/clean/sigma=${SIGMA}" \
    python3 -u scripts/train_apebench_smoke.py \
      --model lemo_pc_nd --regime clean --output_dir $OUT \
      --sigma $SIGMA \
      --family dist_exp_rd_2d \
      --noise_std 0.05 --downsample_factor 2 \
      --epochs 10 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --seed 42 \
      --data_dir data_dde_pde --residual_anchor
done

# -----------------------------------------------------------------------------
# Phase 3: orbit_ood family (only if data already generated).
# 2 cells.
# -----------------------------------------------------------------------------
if [ -d data_orbit_ood/m4 ]; then
  for MODEL in lemo_pc_nd per_lag_mlp_nd; do
    OUT=outputs/audit/${MODEL}_orbit_m4_s42
    run_one "${MODEL}/orbit_m4" \
      python3 -u scripts/train_apebench_smoke.py \
        --family dist_exp_rd_2d_orbit \
        --model $MODEL --regime clean \
        --noise_std 0.0 --downsample_factor 1 \
        --epochs 5 --batch_size 4 \
        --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
        --seed 42 \
        --data_dir data_orbit_ood/m4 \
        --output_dir $OUT --residual_anchor
  done
else
  echo "INFO: data_orbit_ood/m4 not found, skipping orbit audit"
  echo "      (expected if orbit_ood_data sbatch hasn't run yet)"
  echo "SKIP lemo_pc_nd/orbit_m4  REASON=data_not_generated" >> $RESULTS
  echo "SKIP per_lag_mlp_nd/orbit_m4  REASON=data_not_generated" >> $RESULTS
fi

# -----------------------------------------------------------------------------
# Final summary.
# -----------------------------------------------------------------------------
echo ""
echo "================================="
echo "AUDIT SUMMARY ($(date))"
echo "================================="
cat $RESULTS
echo ""
TOTAL=$(wc -l < $RESULTS)
PASS=$(grep -c '^PASS' $RESULTS || echo 0)
FAIL=$(grep -c '^FAIL' $RESULTS || echo 0)
SKIP=$(grep -c '^SKIP' $RESULTS || echo 0)
echo "TOTAL: $TOTAL"
echo "PASS:  $PASS"
echo "FAIL:  $FAIL"
echo "SKIP:  $SKIP"
echo ""
if [ "$FAIL" -gt 0 ]; then
  echo ">>> AUDIT FAILED. The cells above will silently die in the full sweep."
  echo ">>> Do NOT submit / leave running before fixing. Inspect logs:"
  echo ">>>   ls outputs/audit/logs/ | head"
  exit 2
fi
echo ">>> ALL CELLS PASSED. Safe to deploy 386-cell sweep and walk away."

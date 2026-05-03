#!/bin/bash
# Pod 2 recovery — runs detached.
#
# Symptom: data gen for dist_exp_rd_2d + dist_powerlaw_rd_2d hung at 100% CPU
# for 71+ min with zero output, while gauss/gamma/uniform finished normally.
# 5-way parallel gen on Pod 2 over-subscribed the CPU (load avg 2.4, ~2
# effective cores) and the 2 heavier families never made progress.
#
# Recovery steps:
#  1. Kill the 2 stuck gen procs and the parent bootstrap (which is blocked at
#     `wait` and would otherwise proceed to a smoke test that's guaranteed to
#     fail because exp + powerlaw have no shards).
#  2. Clean partial dirs for those two families.
#  3. Re-run gen sequentially (one at a time, full CPU), unbuffered output.
#  4. Re-launch the bootstrap with the same offload range. The bootstrap's
#     gen-presence check skips re-gen since data is now complete.

set -e
cd /workspace/dde-fno
LOG=train_logs/offload/_pod2_recovery.log
mkdir -p train_logs/offload
echo "[recovery] $(date) === Pod 2 recovery start ===" > "$LOG"

# 1. Kill stuck procs + bootstrap.
echo "[recovery] $(date) killing stuck gen + bootstrap..." >> "$LOG"
pkill -f "gen_dde_pde_data\.py.*dist_exp_rd_2d" 2>/dev/null || true
pkill -f "gen_dde_pde_data\.py.*dist_powerlaw_rd_2d" 2>/dev/null || true
pkill -f "_bootstrap_runpod_offload" 2>/dev/null || true
sleep 5
echo "[recovery] $(date) procs after kill:" >> "$LOG"
pgrep -af "gen_dde_pde|_bootstrap_runpod" >> "$LOG" 2>&1 || echo "  (none)" >> "$LOG"

# 2. Clean partial dirs.
echo "[recovery] $(date) cleaning partial data dirs..." >> "$LOG"
rm -rf data_dde_pde/dist_exp_rd_2d data_dde_pde/dist_powerlaw_rd_2d

# 3. Re-run gen for exp (sequential, unbuffered).
echo "[recovery] $(date) running exp gen sequentially with -u..." >> "$LOG"
python3 -u scripts/gen_dde_pde_data.py --family dist_exp_rd_2d \
  --num_train 1000 --num_val 200 --num_test 200 --num_points 64 \
  --T_total 1.28 --dt 0.01 --n_hist 64 --n_out 64 \
  --out_dir data_dde_pde \
  >> "$LOG" 2>&1
EXP_RC=$?
echo "[recovery] $(date) exp gen finished rc=$EXP_RC" >> "$LOG"
if [ $EXP_RC -ne 0 ]; then
  echo "[recovery] $(date) exp gen FAILED — aborting recovery" >> "$LOG"
  exit 1
fi
if [ ! -f data_dde_pde/dist_exp_rd_2d/test/shard_000.npz ]; then
  echo "[recovery] $(date) exp gen finished rc=0 but test/shard_000.npz missing — aborting" >> "$LOG"
  exit 2
fi

# 4. Re-run gen for powerlaw (sequential, unbuffered).
echo "[recovery] $(date) running powerlaw gen sequentially with -u..." >> "$LOG"
python3 -u scripts/gen_dde_pde_data.py --family dist_powerlaw_rd_2d \
  --num_train 1000 --num_val 200 --num_test 200 --num_points 64 \
  --T_total 1.28 --dt 0.01 --n_hist 64 --n_out 64 \
  --out_dir data_dde_pde \
  >> "$LOG" 2>&1
PL_RC=$?
echo "[recovery] $(date) powerlaw gen finished rc=$PL_RC" >> "$LOG"
if [ $PL_RC -ne 0 ]; then
  echo "[recovery] $(date) powerlaw gen FAILED — aborting recovery" >> "$LOG"
  exit 3
fi
if [ ! -f data_dde_pde/dist_powerlaw_rd_2d/test/shard_000.npz ]; then
  echo "[recovery] $(date) powerlaw gen finished rc=0 but test/shard_000.npz missing — aborting" >> "$LOG"
  exit 4
fi

# 5. Re-launch the bootstrap with the original Pod 2 range (cells 46-85, 10 GPUs).
echo "[recovery] $(date) all 5 families now have data — re-launching bootstrap..." >> "$LOG"
setsid nohup bash scripts/_bootstrap_runpod_offload.sh 46 40 10 \
  < /dev/null > train_logs/offload/_bootstrap.log 2>&1 &
BOOT_PID=$!
echo "[recovery] $(date) bootstrap PID $BOOT_PID launched" >> "$LOG"
sleep 5
if kill -0 "$BOOT_PID" 2>/dev/null; then
  echo "[recovery] $(date) bootstrap PID confirmed alive. Recovery DONE." >> "$LOG"
else
  echo "[recovery] $(date) WARNING: bootstrap PID $BOOT_PID not found" >> "$LOG"
fi

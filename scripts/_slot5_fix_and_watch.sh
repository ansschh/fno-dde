#!/bin/bash
# Slot 5 (orbit OOD pod) fix-and-watch.
#
# 1. Patches manifest.json files for m=1,2,4,8,16,32 to add the keys that
#    apebench_dataset.py requires (spatial_shape, spatial_dims, num_channels,
#    n_hist, n_out, params_dim).  The orbit gen wrote skeletal manifests
#    that broke the smoke test with KeyError: 'spatial_shape'.
# 2. Re-launches the orbit bootstrap detached (gen will be skipped, smoke
#    re-runs, sweep launches if smoke passes).
# 3. Watches for sweep progress every 120s for up to 60 minutes.
#    If no test_results.json appears in outputs/orbit_ood_runpod/ within
#    that window, declares Slot 5 unfixable and exits.
#
# Run on Slot 5:
#   setsid nohup bash scripts/_slot5_fix_and_watch.sh \
#     < /dev/null > train_logs/orbit/_slot5_fix_and_watch.log 2>&1 &

cd /workspace/dde-fno
LOG=train_logs/orbit/_slot5_fix_and_watch.log
mkdir -p train_logs/orbit
echo "[slot5-fix] $(date) === Slot 5 fix-and-watch start ===" > "$LOG"

# 1. Patch manifests inline (Python).
python3 - <<'PYEOF' >> "$LOG" 2>&1
import json
from pathlib import Path
keys = {
    "spatial_shape": [64, 64],
    "spatial_dims": 2,
    "num_channels": 1,
    "n_hist": 64,
    "n_out": 64,
    "params_dim": 1,
}
for m in [1, 2, 4, 8, 16, 32]:
    p = Path(f"/workspace/dde-fno/data_orbit_ood/m{m}/dist_exp_rd_2d_orbit/manifest.json")
    if not p.exists():
        print(f"[patch] MISSING manifest for m={m}: {p}")
        continue
    d = json.loads(p.read_text())
    added = []
    for k, v in keys.items():
        if k not in d:
            d[k] = v
            added.append(k)
    p.write_text(json.dumps(d, indent=2))
    print(f"[patch] m={m}: added {added}" if added else f"[patch] m={m}: already complete")
print("[patch] done")
PYEOF

PATCH_RC=$?
echo "[slot5-fix] $(date) patch finished rc=$PATCH_RC" >> "$LOG"
if [ $PATCH_RC -ne 0 ]; then
    echo "[slot5-fix] $(date) PATCH FAILED, exiting" >> "$LOG"
    exit 1
fi

# 2. Re-launch the bootstrap detached.  It will skip data gen (data is on
#    disk), re-run the 5-epoch smoke test (should now pass), and launch
#    the 36-cell sweep on the 7 GPUs.
echo "[slot5-fix] $(date) re-launching bootstrap..." >> "$LOG"
setsid nohup bash scripts/_bootstrap_runpod_orbit.sh \
  < /dev/null \
  > train_logs/orbit/_bootstrap_v2.log 2>&1 &
BOOT_PID=$!
echo "[slot5-fix] $(date) bootstrap PID $BOOT_PID launched" >> "$LOG"

# 3. Watch for progress.  60-min deadline, poll every 120s.
START_TS=$(date +%s)
DEADLINE=$((60 * 60))   # 60 min
echo "[slot5-fix] $(date) watching for sweep progress (60-min deadline)..." >> "$LOG"

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    DONE=$(find /workspace/dde-fno/outputs/orbit_ood_runpod -name test_results.json -size +0 2>/dev/null | wc -l)
    GPU_BUSY=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
    echo "[slot5-fix] $(date) elapsed=${ELAPSED}s done_cells=$DONE gpu_busy_sum=$GPU_BUSY%" >> "$LOG"
    if [ "$DONE" -ge 1 ]; then
        echo "[slot5-fix] $(date) SUCCESS: at least 1 cell completed. Watcher exiting (sweep continues)." >> "$LOG"
        exit 0
    fi
    if [ "$ELAPSED" -ge "$DEADLINE" ]; then
        # Final tail of bootstrap_v2 log to surface the actual failure.
        echo "[slot5-fix] $(date) DEADLINE HIT — sweep did not produce any cells in 60 min." >> "$LOG"
        echo "[slot5-fix] === bootstrap_v2 tail ===" >> "$LOG"
        tail -40 train_logs/orbit/_bootstrap_v2.log >> "$LOG" 2>/dev/null
        echo "[slot5-fix] === smoke log tail ===" >> "$LOG"
        tail -30 train_logs/orbit/_smoke.log >> "$LOG" 2>/dev/null
        echo "[slot5-fix] $(date) Slot 5 declared unfixable. Orbit data preserved at data_orbit_ood/." >> "$LOG"
        exit 2
    fi
    sleep 120
done

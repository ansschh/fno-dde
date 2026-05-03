#!/bin/bash
# Rescue waiter — runs on Pod A in the background.
#
# Polls until the main offload launcher and all training subprocesses exit,
# then re-launches the launcher with the SAME cell range. The idempotent
# dispatcher (_run_offload_cell.py) will skip cells with existing
# test_results.json, so only previously-failed cells (e.g. the ~22 ffno_nd
# cells that crashed before the bug fix) get re-trained.
#
# Usage on Pod A: launched detached after deploying this commit:
#   setsid nohup bash scripts/_rescue_waiter.sh \
#     < /dev/null > train_logs/offload/_rescue_waiter_launch.log 2>&1 &
#
# Configuration: re-runs Pod A's range 223..272 (50 cells, NGPU=7).

set -e
cd /workspace/dde-fno
LOG=train_logs/offload/_rescue_waiter.log
mkdir -p train_logs/offload

START=223
COUNT=50
NGPU=7

echo "[rescue-waiter] $(date) start. Polling main launcher (every 120s)..." > "$LOG"

# Initial small delay so we don't false-positive before pgrep observes the main.
sleep 30

while true; do
  if ! pgrep -af "_launch_caltech_offload\.sh|train_apebench_smoke\.py" > /dev/null 2>&1; then
    # Verify with a 30s recheck — guards against transient process-table gaps
    # between cells.
    echo "[rescue-waiter] $(date) main appears idle, rechecking after 30s..." >> "$LOG"
    sleep 30
    if ! pgrep -af "_launch_caltech_offload\.sh|train_apebench_smoke\.py" > /dev/null 2>&1; then
      break
    fi
  fi
  N=$(pgrep -af "_launch_caltech_offload\.sh|train_apebench_smoke\.py" 2>/dev/null | wc -l)
  echo "[rescue-waiter] $(date) main still active ($N procs)" >> "$LOG"
  sleep 120
done

echo "[rescue-waiter] $(date) main done. Counting completed cells before rescue..." >> "$LOG"
DONE=$(find outputs -name test_results.json -size +0 2>/dev/null | wc -l)
echo "[rescue-waiter] cells with non-empty test_results.json on pod: $DONE" >> "$LOG"

echo "[rescue-waiter] $(date) launching rescue: START=$START COUNT=$COUNT NGPU=$NGPU" >> "$LOG"
setsid bash scripts/_launch_caltech_offload.sh "$START" "$COUNT" "$NGPU" \
  < /dev/null \
  > train_logs/offload/_master_rescue.log 2>&1 &
RESCUE_PID=$!
echo "[rescue-waiter] $(date) rescue launched, PID $RESCUE_PID" >> "$LOG"
sleep 5
if kill -0 "$RESCUE_PID" 2>/dev/null; then
  echo "[rescue-waiter] $(date) rescue PID $RESCUE_PID confirmed alive. Exiting waiter." >> "$LOG"
else
  echo "[rescue-waiter] $(date) WARNING: rescue PID $RESCUE_PID not found 5s after launch." >> "$LOG"
fi

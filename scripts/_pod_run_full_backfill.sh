#!/usr/bin/env bash
# Full backfill: run ALL post-hoc evals on every checkpoint under the given
# roots. Sharded across 8 GPUs. Each step is idempotent.
#
#   1. _pod_unified_worker.py  — capture, lipschitz, equivariance_dense,
#                                 adversarial_dense, noise_dense, per_frame
#   2. eval_cross_family.py    — cross_family_relL2.json
#   3. eval_long_horizon.py    — long_horizon.json (T={64,128,256,512})
#
# Usage:
#   bash scripts/_pod_run_full_backfill.sh ROOT1 ROOT2 ...
#
# Output goes to a tmux session "backfill" — attach with `tmux attach -t backfill`.
set -euo pipefail

cd /workspace/dde-fno

if [[ $# -lt 1 ]]; then
    ROOTS=("/workspace/dde-fno/extracted" "/workspace/dde-fno/outputs")
else
    ROOTS=("$@")
fi

DATA_DIR="${DATA_DIR:-data_dde_pde}"
SHARDS_DIR="/tmp/backfill_shards"
LOG_DIR="/workspace/dde-fno/outputs/_backfill_logs/$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$SHARDS_DIR" "$LOG_DIR"

echo "[backfill] roots: ${ROOTS[*]}"
echo "[backfill] data: $DATA_DIR"
echo "[backfill] logs: $LOG_DIR"

# Build the unified ckpt list.
ALL_CKPTS="$SHARDS_DIR/all_ckpts.txt"
> "$ALL_CKPTS"
for R in "${ROOTS[@]}"; do
    if [[ -d "$R" ]]; then
        find "$R" -name 'best_model.pt' 2>/dev/null >> "$ALL_CKPTS"
    fi
done
TOTAL=$(wc -l < "$ALL_CKPTS")
echo "[backfill] $TOTAL checkpoints"

if [[ "$TOTAL" -eq 0 ]]; then
    echo "[backfill] no checkpoints; aborting"
    exit 1
fi

# Shard by hash so each GPU gets a deterministic slice.
for g in 0 1 2 3 4 5 6 7; do
    awk -v g=$g 'NR % 8 == g' "$ALL_CKPTS" > "$SHARDS_DIR/shard_$g.txt"
    n=$(wc -l < "$SHARDS_DIR/shard_$g.txt")
    echo "[backfill] shard_$g: $n cells"
done

# Launch one worker per GPU.
SESSION="backfill"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n master "echo backfill master; sleep 86400"

for g in 0 1 2 3 4 5 6 7; do
    SHARD="$SHARDS_DIR/shard_$g.txt"
    LOG="$LOG_DIR/gpu_$g.log"

    CMD=$(cat <<EOF
set -e
cd /workspace/dde-fno
echo '[gpu $g] === step1: unified worker (capture/lipschitz/equiv/adv/noise) ===' | tee -a $LOG
CUDA_VISIBLE_DEVICES=$g python3 scripts/_pod_unified_worker.py \\
    --shard $SHARD --data_dir $DATA_DIR --gpu 0 2>&1 | tee -a $LOG

echo '[gpu $g] === step2: cross_family ===' | tee -a $LOG
CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_cross_family.py \\
    --shard $SHARD --data_dir $DATA_DIR --gpu 0 --n_batches 8 2>&1 | tee -a $LOG

echo '[gpu $g] === step3: long_horizon ===' | tee -a $LOG
CUDA_VISIBLE_DEVICES=$g python3 scripts/eval_long_horizon.py \\
    --shard $SHARD --data_dir $DATA_DIR 2>&1 | tee -a $LOG

echo '[gpu $g] DONE' | tee -a $LOG
EOF
)
    tmux new-window -t "$SESSION" -n "g$g" "bash -c '$CMD'"
done

echo "[backfill] launched 8 workers in tmux session '$SESSION'"
echo "[backfill] monitor: ssh ... 'tmux attach -t $SESSION'"
echo "[backfill] tail logs: ssh ... 'tail -f $LOG_DIR/gpu_*.log'"

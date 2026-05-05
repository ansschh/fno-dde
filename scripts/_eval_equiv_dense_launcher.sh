#!/bin/bash
# Parallel eval_equivariance_dense launcher for 8 GPUs.
#
# Shards by (family, regime). Each GPU runs eval_equivariance_dense.py
# on a single (fam, reg) tuple, processing all ckpts at that path
# sequentially. With 5 fams x 3 regimes = 15 shards on 8 GPUs:
# ~2 sequential rounds, ~15 min/round = ~30 min total.
#
# Usage on pod:
#   cd /workspace/dde-fno && bash scripts/_eval_equiv_dense_launcher.sh

set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

N_GPU=${N_GPU:-8}
LOG_DIR=$REPO/train_logs/eval_equiv_dense
mkdir -p "$LOG_DIR"

ROOTS=(
    "$REPO/outputs/a_fix_runpod"
    "$REPO/outputs/film_ablation_runpod"
    "$REPO/outputs/w11_compute_matched_runpod"
)

FAMS=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
REGIMES=(clean lowres noisy)

# Build (fam, regime) shard list
declare -a SHARDS
i=0
for fam in "${FAMS[@]}"; do
    for reg in "${REGIMES[@]}"; do
        SHARDS[$i]="$fam $reg"
        i=$((i+1))
    done
done
TOTAL=${#SHARDS[@]}
echo "[eval-equiv-dense] $TOTAL shards (5 fams x 3 regimes), $N_GPU GPUs"

run_shard() {
    local idx=$1
    local gpu=$2
    read fam reg <<< "${SHARDS[$idx]}"
    local log="$LOG_DIR/shard_$(printf '%02d' $idx)_${fam}_${reg}.log"
    echo "[gpu$gpu] shard $idx: $fam/$reg -> $log" >&2
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval_equivariance_dense.py \
        --roots "${ROOTS[@]}" \
        --data_dir data_dde_pde \
        --families "$fam" \
        --regimes "$reg" \
        --n_batches 16 \
        > "$log" 2>&1
}

# Launch in waves of N_GPU
idx=0
while [ $idx -lt $TOTAL ]; do
    pids=()
    for gpu in $(seq 0 $((N_GPU - 1))); do
        if [ $idx -ge $TOTAL ]; then break; fi
        run_shard $idx $gpu &
        pids+=($!)
        idx=$((idx + 1))
    done
    wait "${pids[@]}"
done
echo "[eval-equiv-dense] all $TOTAL shards complete"

#!/bin/bash
# Aggressive 200-epoch retrain across multi-GPU pod (Pod 1 7×A100 OR Pod 2 7×4090).
# Splits cells across GPUs cyclically; each GPU processes its assigned cells
# sequentially. Cells assigned by --start / --count flags so we can split across
# pods (Pod 1 starts at 0, Pod 2 starts at 23 or 24 — adjust per pod).
#
# Usage: bash _launch_full_retrain.sh <START_INDEX> <NUM_CELLS>
#   e.g.,  bash _launch_full_retrain.sh 0 23   (Pod 1)
#          bash _launch_full_retrain.sh 23 22  (Pod 2)
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-23}
NGPU=7
EPOCHS=200

# 45-cell list: 5 fams × 3 regimes × 3 seeds (clean → lowres → noisy, then by seed)
FAMS=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
REGIMES=(clean lowres noisy)
SEEDS=(42 123 456)

# Generate the full cell list
CELLS=()
for fam in "${FAMS[@]}"; do
  for reg in "${REGIMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      CELLS+=("$fam:$reg:$seed")
    done
  done
done

mkdir -p train_logs/retrain
END=$((START + COUNT))
echo "Pod will run cells $START..$((END-1)) of ${#CELLS[@]} total"

# Distribute cells across GPUs cyclically; each GPU runs its cells sequentially
for gpu in $(seq 0 $((NGPU-1))); do
  (
    for i in $(seq $START $((END-1))); do
      if [ $((i % NGPU)) -ne $gpu ]; then continue; fi
      IFS=':' read -r fam reg seed <<< "${CELLS[$i]}"
      log=train_logs/retrain/${fam}_${reg}_s${seed}.log
      echo "[GPU $gpu] starting cell $i: $fam $reg s$seed -> $log"
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 \
        python3 -u scripts/train_apebench_smoke.py \
          --family $fam --model lemo_pc_nd --regime $reg \
          --noise_std 0.05 --downsample_factor 2 \
          --epochs $EPOCHS --batch_size 4 \
          --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
          --seed $seed \
          --data_dir data_dde_pde \
          --output_dir outputs/film_fix_full \
          --residual_anchor \
          > $log 2>&1
      echo "[GPU $gpu] done cell $i: $fam $reg s$seed"
    done
  ) &
done
wait
echo "All cells $START..$((END-1)) done."

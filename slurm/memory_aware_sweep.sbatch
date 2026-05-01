#!/bin/bash
#SBATCH --job-name=lemo-memory-aware
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-134        # 135 cells: 3 models * 5 fams * 3 regimes * 3 seeds
#SBATCH --output=slurm_logs/%A_%a_mema.out
#SBATCH --error=slurm_logs/%A_%a_mema.err

# Memory-aware baselines (T4 plus NDDE plus S4): NIDE_ND, NDDE_ND, S4_ND.
# Closes Round 3 fatal #1 (memory-aware head-to-head).
#
# 135 cells = 3 models * 5 dist-kernel families * 3 regimes * 3 seeds.
# Matched compute envelope to LEMO-PC headline.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

MODELS=(nide_nd ndde_nd s4_nd)
FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
REGIMES=(clean lowres noisy)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
n_seeds=${#SEEDS[@]}
n_regimes=${#REGIMES[@]}
n_fams=${#FAMILIES[@]}

seed_idx=$(( idx % n_seeds ))
regime_idx=$(( (idx / n_seeds) % n_regimes ))
fam_idx=$(( (idx / (n_seeds * n_regimes)) % n_fams ))
model_idx=$(( idx / (n_seeds * n_regimes * n_fams) ))

MODEL=${MODELS[$model_idx]}
FAM=${FAMILIES[$fam_idx]}
REGIME=${REGIMES[$regime_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[$(date)] Job $idx: model=$MODEL family=$FAM regime=$REGIME seed=$SEED"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/train_apebench_smoke.py \
  --family $FAM --model $MODEL --regime $REGIME \
  --noise_std 0.05 --downsample_factor 2 \
  --epochs 200 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed $SEED \
  --data_dir data_dde_pde \
  --output_dir "outputs/memory_aware/raw/${FAM}/${REGIME}/${MODEL}/s${SEED}" \
  --residual_anchor

echo "[$(date)] Job $idx training done -- running post-hoc captures..."

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/capture_paper_artifacts.py \
  --layer_root outputs/memory_aware \
  --data_dir data_dde_pde \
  --device cuda || echo "WARN: capture failed for cell $idx"

echo "[$(date)] Job $idx done."

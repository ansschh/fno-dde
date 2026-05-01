#!/bin/bash
#SBATCH --job-name=lemo-orbit-ood
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-35         # 36 cells: 2 models * 6 m-values * 3 seeds
#SBATCH --output=slurm_logs/%A_%a_orbit.out
#SBATCH --error=slurm_logs/%A_%a_orbit.err

# T2: lag-shift orbit OOD (Theorem 3.4 + Cor C12 augmentation covering-radius).
# Tests whether equivariant LEMO-PC has orbit-constant test error vs the
# non-equivariant per_lag_mlp_nd that decays as ~1/m.
#
# 36 cells = 2 models * 6 m-values * 3 seeds.
# PRECONDITION: data_orbit_ood/m{1,2,4,8,16,32}/ must exist (run
# slurm/orbit_ood_data.sbatch first).

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

MODELS=(lemo_pc_nd per_lag_mlp_nd)
M_VALUES=(1 2 4 8 16 32)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
n_seeds=${#SEEDS[@]}      # 3
n_m=${#M_VALUES[@]}       # 6
# n_models = 2

seed_idx=$(( idx % n_seeds ))
m_idx=$(( (idx / n_seeds) % n_m ))
model_idx=$(( idx / (n_seeds * n_m) ))

MODEL=${MODELS[$model_idx]}
M=${M_VALUES[$m_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[$(date)] Job $idx: model=$MODEL m=$M seed=$SEED"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/train_apebench_smoke.py \
  --family dist_exp_rd_2d_orbit --model $MODEL --regime clean \
  --noise_std 0.0 --downsample_factor 1 \
  --epochs 200 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed $SEED \
  --data_dir "data_orbit_ood/m${M}" \
  --output_dir "outputs/orbit_ood/${MODEL}/m${M}/s${SEED}" \
  --residual_anchor

echo "[$(date)] Job $idx done."

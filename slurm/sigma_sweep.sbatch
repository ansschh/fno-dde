#!/bin/bash
#SBATCH --job-name=lemo-sigma-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-59         # 60 cells: 4 sigma * 5 fams * 3 seeds
#SBATCH --output=slurm_logs/%A_%a_sigma.out
#SBATCH --error=slurm_logs/%A_%a_sigma.err

# σ-stability sweep on Caltech HPC.
# 60 cells: σ ∈ {0.5, 0.7, 0.9, 0.99} × 5 dist-kernel families × 3 seeds × 1 regime (clean)
# Each cell: 200 epochs, batch=4, width=64, lag_modes=24, residual_anchor.
# Per-cell wall time on A100/V100: ~3-4h. SLURM gives us 12h per cell — plenty.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

# Build the (sigma, family, seed) job grid.
SIGMAS=(0.5 0.7 0.9 0.99)
FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
n_seeds=${#SEEDS[@]}
n_fams=${#FAMILIES[@]}

seed_idx=$(( idx % n_seeds ))
fam_idx=$(( (idx / n_seeds) % n_fams ))
sigma_idx=$(( idx / (n_seeds * n_fams) ))

SIGMA=${SIGMAS[$sigma_idx]}
FAM=${FAMILIES[$fam_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[$(date)] Job $idx: sigma=$SIGMA family=$FAM seed=$SEED"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/train_apebench_smoke.py \
  --family $FAM --model lemo_pc_nd --regime clean \
  --noise_std 0.05 --downsample_factor 2 \
  --epochs 200 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed $SEED \
  --data_dir data_dde_pde \
  --output_dir "outputs/sigma_${SIGMA}/raw/${FAM}/clean/lemo_pc_nd/s${SEED}" \
  --residual_anchor --sigma $SIGMA

echo "[$(date)] Job $idx done"

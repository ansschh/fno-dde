#!/bin/bash
#SBATCH --job-name=lemo-film-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-134        # 135 cells: 3 models * 5 fams * 3 regimes * 3 seeds
#SBATCH --output=slurm_logs/%A_%a_film.out
#SBATCH --error=slurm_logs/%A_%a_film.err

# T1: FiLM-vs-equivariance ablation (3 models, NOT 4 -- A4=lemo_pc_nd is
# reused from the headline 45-cell sweep, so we skip the 45 redundant cells).
#
# 135 cells = 3 models * 5 dist-kernel families * 3 regimes * 3 seeds.
# Matched compute envelope to LEMO-PC headline:
#   width=64, n_layers=3, lag_modes=24, spatial_modes=12, batch=4, 200 epochs,
#   residual_anchor=on, sigma=none.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
mkdir -p slurm_logs
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

MODELS=(fno_film_nd noneq_film_nd lemo_bcorrect_nd)
FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
REGIMES=(clean lowres noisy)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
n_seeds=${#SEEDS[@]}      # 3
n_regimes=${#REGIMES[@]}  # 3
n_fams=${#FAMILIES[@]}    # 5
# n_models = 3

seed_idx=$(( idx % n_seeds ))
regime_idx=$(( (idx / n_seeds) % n_regimes ))
fam_idx=$(( (idx / (n_seeds * n_regimes)) % n_fams ))
model_idx=$(( idx / (n_seeds * n_regimes * n_fams) ))

MODEL=${MODELS[$model_idx]}
FAM=${FAMILIES[$fam_idx]}
REGIME=${REGIMES[$regime_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[$(date)] Job $idx: model=$MODEL family=$FAM regime=$REGIME seed=$SEED"

# Skip if this cell already completed (keeps idempotent resubmits cheap).
RESULT_PATH="outputs/film_ablation/raw/${FAM}/${REGIME}/${MODEL}/s${SEED}/test_results.json"
if [ -f "$RESULT_PATH" ]; then
  echo "[$(date)] Job $idx: SKIP (test_results.json already present at $RESULT_PATH)"
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/train_apebench_smoke.py \
  --family $FAM --model $MODEL --regime $REGIME \
  --noise_std 0.05 --downsample_factor 2 \
  --epochs 200 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed $SEED \
  --data_dir data_dde_pde \
  --output_dir "outputs/film_ablation/raw" \
  --residual_anchor

# Post-hoc captures disabled to fit Caltech disk quota.
# Run scripts/capture_paper_artifacts.py locally on downloaded checkpoints instead.

echo "[$(date)] Job $idx done."

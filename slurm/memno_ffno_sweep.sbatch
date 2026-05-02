#!/bin/bash
#SBATCH --job-name=lemo-memno-ffno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-59          # 60 cells: 2 models * 5 fams * 2 regimes * 3 seeds
#SBATCH --output=slurm_logs/%A_%a_mnff.out
#SBATCH --error=slurm_logs/%A_%a_mnff.err

# Memory-aware baselines: MemNO + F-FNO (migrated from RunPod).
# Closes Round 3 fatal #1 (memory-aware head-to-head, MemNO/F-FNO subset).
# Note: clean regime is not included here -- it's covered elsewhere (headline
# sweep reuses these models on clean). This sweep adds lowres + noisy only.
#
# 60 cells = 2 models * 5 dist-kernel families * 2 regimes (lowres, noisy) * 3 seeds.
# Matched compute envelope to LEMO-PC headline.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
mkdir -p slurm_logs
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

MODELS=(memno_nd ffno_nd)
FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
REGIMES=(lowres noisy)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
n_seeds=${#SEEDS[@]}      # 3
n_regimes=${#REGIMES[@]}  # 2
n_fams=${#FAMILIES[@]}    # 5
# n_models = 2

seed_idx=$(( idx % n_seeds ))
regime_idx=$(( (idx / n_seeds) % n_regimes ))
fam_idx=$(( (idx / (n_seeds * n_regimes)) % n_fams ))
model_idx=$(( idx / (n_seeds * n_regimes * n_fams) ))

MODEL=${MODELS[$model_idx]}
FAM=${FAMILIES[$fam_idx]}
REGIME=${REGIMES[$regime_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[$(date)] Job $idx: model=$MODEL family=$FAM regime=$REGIME seed=$SEED"

RESULT_PATH="outputs/memno_ffno/raw/${FAM}/${REGIME}/${MODEL}/s${SEED}/test_results.json"
if [ -f "$RESULT_PATH" ]; then
  echo "[$(date)] Job $idx: SKIP (test_results.json already present)"
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
  --output_dir "outputs/memno_ffno/raw" \
  --residual_anchor

# Post-hoc captures disabled to fit Caltech disk quota.
# Run scripts/capture_paper_artifacts.py locally on downloaded checkpoints instead.

echo "[$(date)] Job $idx done."

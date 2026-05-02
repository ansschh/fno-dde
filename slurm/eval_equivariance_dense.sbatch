#!/bin/bash
#SBATCH --job-name=lemo-equiv-dense
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4              # 5 array tasks, one per family
#SBATCH --output=slurm_logs/%A_%a_equiv.out
#SBATCH --error=slurm_logs/%A_%a_equiv.err

# Dense-k equivariance eval across all checkpoints, for ALL models.
# k in {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64} (11 shifts; previous F08 had only 3).
#
# Per array task: one family. Crawls every sweep root with checkpoints and runs
# eval_equivariance_dense.py, writing equivariance_dense.json next to each best_model.pt.
# Resumable via skip-if-exists.
#
# Sweep roots covered (Caltech filesystem layout):
#   outputs/dist_kernel_v2_p1                  (lemo_pc_nd, 45 ckpts)
#   outputs/film_ablation_caltech              (fno_film_nd, fno_nd, lemo_bcorrect_nd, ...)
#   outputs/final_baselines                    (ffno_nd, memno_nd, 30 ckpts)
#   outputs/memory_aware_caltech               (NIDE / NDDE / per_lag_mlp_nd / ...)
#   outputs/memno_ffno_caltech                 (memno + ffno noisy/lowres expansion)
#
# Estimated time per cell on H200 (forward-pass only, 11 k * 16 batches * 8 samples):
#   ~30-60s including model load + dataset load.
# Per family: ~30 cells (5-9 models * 3 seeds, clean only) * 45s = ~22 min.
# Wall time budget 4h = 240 min — plenty of slack for noisy/lowres if expanded.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
mkdir -p slurm_logs
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm tensorboard --break-system-packages -q

FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
FAM=${FAMILIES[$SLURM_ARRAY_TASK_ID]}

# All sweep roots that may contain checkpoints. Missing dirs are silently skipped
# by the python script (it only globs raw/**/best_model.pt).
ROOTS=(
  "outputs/dist_kernel_v2_p1"
  "outputs/film_ablation_caltech"
  "outputs/final_baselines"
  "outputs/memory_aware_caltech"
  "outputs/memno_ffno_caltech"
  "outputs/film_ablation"
  "outputs/memory_aware"
  "outputs/memno_ffno"
)

echo "[$(date)] equiv-dense for family=$FAM"
echo "         roots: ${ROOTS[*]}"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/eval_equivariance_dense.py \
  --roots "${ROOTS[@]}" \
  --data_dir data_dde_pde \
  --families "$FAM" \
  --regimes clean \
  --n_batches 16 \
  --shifts "1,2,4,6,8,12,16,24,32,48,64"

echo "[$(date)] family=$FAM done."

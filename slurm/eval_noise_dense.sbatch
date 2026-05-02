#!/bin/bash
#SBATCH --job-name=lemo-noise-dense
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4              # 5 array tasks, one per family
#SBATCH --output=slurm_logs/%A_%a_noise.out
#SBATCH --error=slurm_logs/%A_%a_noise.err

# Dense-sigma Gaussian-noise eval across all checkpoints, for ALL models.
# σ in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0} (9 values).
#
# Per-cell wall time on H200: ~25-50s (forward-only, no autograd needed).
# Per family (~30 clean cells when sweeps land): ~18 min.
# All 5 array tasks parallel: ~20 min wall clock.

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

ROOTS=(
  "outputs/dist_kernel_v2_p1"
  "outputs/film_ablation_caltech"
  "outputs/final_baselines"
  "outputs/memory_aware_caltech"
  "outputs/memno_ffno_caltech"
)

echo "[$(date)] noise-dense for family=$FAM"
echo "         roots: ${ROOTS[*]}"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python3 -u scripts/eval_noise_dense.py \
  --roots "${ROOTS[@]}" \
  --data_dir data_dde_pde \
  --families "$FAM" \
  --regimes clean \
  --n_batches 16 \
  --sigmas "0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0"

echo "[$(date)] family=$FAM done."

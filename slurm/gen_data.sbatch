#!/bin/bash
#SBATCH --job-name=lemo-datagen
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/datagen/%A_%a.out
#SBATCH --error=logs/datagen/%A_%a.err

# Generate the 5 dist-kernel reaction-diffusion 2D families on a Caltech
# compute node (DO NOT run gen_dde_pde_data.py on a login node — the login
# node killer will terminate it).
#
# Adjust --partition above if "expansion" is not the right CPU partition on
# your cluster.  Other common names: cpu, batch, any, gpu (any partition
# works since this is CPU-only; gpu just wastes GPU allocation).

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true

cd $HOME/dde-fno
source venv/bin/activate
mkdir -p logs/datagen data_dde_pde

FAMILIES=(dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d)
FAM=${FAMILIES[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: generating $FAM"
python3 -u scripts/gen_dde_pde_data.py --family $FAM \
  --num_train 1000 --num_val 200 --num_test 200 --num_points 64 \
  --T_total 1.28 --dt 0.01 --n_hist 64 --n_out 64 --out_dir data_dde_pde
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: $FAM done"

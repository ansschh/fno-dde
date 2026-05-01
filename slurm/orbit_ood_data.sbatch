#!/bin/bash
#SBATCH --job-name=lemo-orbit-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_orbit_data.out
#SBATCH --error=slurm_logs/%j_orbit_data.err

# NOTE: this is a CPU-bound data-gen job that uses --partition=gpu --gres=gpu:1
# only because Caltech HPC doesn't expose a `cpu` partition under that name.
# The Python script does not actually use CUDA. Wasteful by ~30min of GPU time.
# If you find the right CPU-only partition name (sinfo -s), update this.

# T2 orbit-OOD data generation (CPU job).
# Produces data_orbit_ood/m{1,2,4,8,16,32}/ datasets at the audited config:
#   tau=0.25, A in [8,11], IC amplitude 0.45 -- pairwise rel-L2 between
#   cyclic shifts max 0.156 (vs 0.005 baseline). PAC gap will be observable.
#
# Submit this BEFORE orbit_ood.sbatch.  Once it finishes, launch the GPU
# sweep with sbatch slurm/orbit_ood.sbatch.

set -e

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true

REPO=$HOME/dde-fno
cd $REPO
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
pip install h5py matplotlib scipy pyyaml tqdm --break-system-packages -q

echo "[$(date)] Phase 0: orbit-OOD AUDIT (single orbit sanity check)"
python3 scripts/gen_orbit_ood_data.py --audit_only \
  --tau 0.25 --tau_max 0.4 --dt 0.01 \
  --n_hist 64 --n_out 64 --n_grid 64 \
  --out_dir data_orbit_ood

echo "[$(date)] Phase 1: orbit-OOD data generation (256 orbits + 32 val per m)"
python3 scripts/gen_orbit_ood_data.py \
  --num_orbits 256 --num_val_orbits 32 \
  --m_values 1,2,4,8,16,32 \
  --n_hist 64 --n_out 64 --n_grid 64 \
  --dt 0.01 --tau 0.25 --tau_max 0.4 \
  --out_dir data_orbit_ood --seed 42

echo "[$(date)] Orbit-OOD data generation complete."
ls -la data_orbit_ood/

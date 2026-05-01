#!/usr/bin/env bash
# Run on Caltech HPC AFTER deploy_to_caltech.sh completed.
#
# This:
#   1. Sets up the Python venv with torch + h5py + matplotlib
#   2. Submits the σ-stability SLURM array job (60 cells, 4 σ × 5 fams × 3 seeds)
#   3. Prints monitoring commands
#
# Total compute: ~60 cell-jobs × ~3-4h on a single GPU each.  Caltech HPC's
# scheduler will run them in parallel up to your fair-share allocation.

set -e
cd ~/dde-fno

echo "=== STAGE 1: Python env setup ==="
module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
  pip install torch torchvision -q
pip install h5py matplotlib scipy pyyaml tqdm tensorboard numpy -q

echo ""
echo "=== STAGE 2: smoke check ==="
python3 -c "
import sys; sys.path.insert(0, 'src')
from train.build_model import build_model
import torch
cfg = {'model_class': 'lemo_pc_nd', 'model': {'spatial_shape': (32, 32), 'spatial_modes': (12, 12), 'lag_modes': 24, 'width': 64, 'n_layers': 3, 'params_dim': 1, 'sigma': 0.5}}
m = build_model(cfg, in_channels=4, out_channels=1, length=128)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('  build LEMO_PC sigma=0.5 OK, params=', n)
"

echo ""
echo "=== STAGE 3: submit σ-stability sweep (60 cells) ==="
mkdir -p slurm_logs
JOB_ID=$(sbatch --parsable slurm/sigma_sweep.sbatch)
echo "Submitted SLURM array job: $JOB_ID (60 tasks)"

echo ""
echo "=== MONITORING COMMANDS ==="
echo "  Check status:        squeue -u \$USER"
echo "  Job details:         sacct -j ${JOB_ID}"
echo "  Live logs:           tail -f slurm_logs/${JOB_ID}_*.out"
echo "  Count completed:     find outputs/sigma_*/raw -name test_results.json | wc -l"
echo ""
echo "When all 60 finish, bundle results:"
echo "  tar czf sigma_sweep_results.tar.gz outputs/sigma_*/"
echo "  scp caltech-hpc:~/dde-fno/sigma_sweep_results.tar.gz <local_path>"

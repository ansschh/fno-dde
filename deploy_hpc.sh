#!/bin/bash
# =============================================================
# DDE-FNO Hard Benchmark: HPC Deployment Script
# =============================================================
# Run this AFTER you've cloned/copied the repo to the HPC.
#
# Usage:
#   ssh <username>@login.hpc.caltech.edu   # (Duo 2FA)
#   cd /path/to/dde-fno
#   bash deploy_hpc.sh
# =============================================================

set -e

echo "============================================================="
echo "  DDE-FNO Hard Benchmark — HPC Deployment"
echo "============================================================="

# --- 0. Detect environment ---
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
echo "[0] Working directory: $REPO_ROOT"

# --- 1. Setup Python environment ---
echo ""
echo "[1] Setting up Python environment..."

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

if [ ! -d "venv" ]; then
    echo "    Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "    Installing dependencies..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    pip install torch torchvision -q
pip install numpy scipy matplotlib pyyaml h5py tensorboard tqdm -q

echo "    Python: $(python3 --version)"
echo "    PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "    CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# --- 2. Create directories ---
echo ""
echo "[2] Creating directories..."
mkdir -p data slurm_logs reports/pilot_results reports/sweep_results
mkdir -p reports/hard_benchmark_figures

# --- 3. Quick smoke test (CPU, ~2 min) ---
echo ""
echo "[3] Running smoke test (CPU, small data)..."
python3 scripts/integration_test.py --device cpu --n-samples 50 --epochs 5
echo "    Smoke test passed!"

# --- 4. Pilot runs (submit as SLURM job) ---
echo ""
echo "[4] Submitting pilot jobs..."

# Write a pilot batch script
cat > slurm_logs/pilot_all.sbatch << 'PILOT_EOF'
#!/bin/bash
#SBATCH --job-name=dde-pilot-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j_pilot_all.out
#SBATCH --error=slurm_logs/%j_pilot_all.err

module load cuda python/3.10
source venv/bin/activate
cd $SLURM_SUBMIT_DIR

echo "=== Piloting all DDE families ==="
python3 scripts/run_pilot.py --all-dde \
    --n_train 1600 --n_val 200 --n_test 200 \
    --epochs 50 --device cuda \
    --save-results reports/pilot_results/pilot_dde.json

echo ""
echo "=== Piloting all PDE families ==="
python3 scripts/run_pilot.py --all-pde \
    --n_train 1600 --n_val 200 --n_test 200 \
    --epochs 50 --device cuda \
    --save-results reports/pilot_results/pilot_pde.json

echo ""
echo "=== Generating figures from pilot results ==="
python3 scripts/benchmark_pack/generate_hard_benchmark_figures.py \
    --pilot_results reports/pilot_results/pilot_dde.json \
    --out_dir reports/hard_benchmark_figures

echo "Done!"
PILOT_EOF

PILOT_JOB=$(sbatch slurm_logs/pilot_all.sbatch | awk '{print $4}')
echo "    Pilot job submitted: $PILOT_JOB"

# --- 5. Full benchmark sweep (after pilots) ---
echo ""
echo "[5] Submitting full benchmark sweep (depends on pilot)..."

# Stage 1: Data generation (CPU jobs, no GPU needed)
echo "    [5a] Data generation jobs..."
python3 slurm/sweep.py configs/sweep_hard_benchmark.yaml \
    --stage generate --data_dir data 2>&1 | tail -5

# Stage 2: Training (GPU jobs, depend on data gen)
echo ""
echo "    [5b] Training jobs..."
python3 slurm/sweep.py configs/sweep_hard_benchmark.yaml \
    --stage train --data_dir data 2>&1 | tail -5

# Stage 3: Evaluation (GPU jobs, depend on training)
echo ""
echo "    [5c] Evaluation jobs..."
python3 slurm/sweep.py configs/sweep_hard_benchmark.yaml \
    --stage eval --data_dir data 2>&1 | tail -5

# --- 6. Summary ---
echo ""
echo "============================================================="
echo "  DEPLOYMENT COMPLETE"
echo "============================================================="
echo ""
echo "  Jobs submitted. Monitor with:"
echo "    squeue -u \$USER"
echo "    sacct -j <jobid>"
echo ""
echo "  Check pilot results:"
echo "    cat reports/pilot_results/pilot_dde.json"
echo "    cat reports/pilot_results/pilot_pde.json"
echo ""
echo "  After sweep completes, generate figures:"
echo "    python3 scripts/benchmark_pack/generate_hard_benchmark_figures.py \\"
echo "      --results_dir reports/sweep_results"
echo ""
echo "  Sweep manifest:"
echo "    cat slurm_logs/sweep_manifest_hard_benchmark_v1.json"
echo "============================================================="

#!/bin/bash
#SBATCH --job-name=dde-gen-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Detect DDE vs PDE family and call the correct generator
PDE_FAMILIES="burgers ks helmholtz wave"
FAMILY="{family}"

if echo "$PDE_FAMILIES" | grep -qw "$FAMILY"; then
    echo "Generating PDE data for $FAMILY..."
    python3 src/datasets/generate_pde.py "$FAMILY" \
        --n_train={n_train} --n_val={n_val} --n_test={n_test} \
        --output_dir={data_dir} --seed={seed}
else
    echo "Generating DDE data for $FAMILY..."
    python3 src/datasets/generate_python.py "$FAMILY" \
        --n_train={n_train} --n_val={n_val} --n_test={n_test} \
        --output_dir={data_dir} --seed={seed}
fi

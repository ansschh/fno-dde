#!/bin/bash
#SBATCH --job-name=dde-eval-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python3 src/benchmarks/eval_comprehensive.py \
    --checkpoint={checkpoint_path} \
    --family={family} \
    --data_dir={data_dir} \
    --output_dir={eval_dir}

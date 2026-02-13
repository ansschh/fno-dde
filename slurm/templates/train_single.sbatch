#!/bin/bash
#SBATCH --job-name=dde-train-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python3 src/train/train_fno_sharded.py \
    --config={config_path} \
    --data_dir={data_dir} \
    --output_dir={output_dir} \
    --device=cuda \
    --seed={seed}

#!/bin/bash
#SBATCH --job-name=dde-train-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load cuda python/3.10

cd $SLURM_SUBMIT_DIR
python src/train/train_fno_sharded.py \
    --config={config_path} \
    --data_dir={data_dir} \
    --output_dir={output_dir} \
    --device=cuda \
    --seed={seed}

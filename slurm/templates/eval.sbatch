#!/bin/bash
#SBATCH --job-name=dde-eval-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load cuda python/3.10

cd $SLURM_SUBMIT_DIR
python src/benchmarks/eval_comprehensive.py \
    --checkpoint={checkpoint_path} \
    --family={family} \
    --data_dir={data_dir} \
    --output_dir={eval_dir}

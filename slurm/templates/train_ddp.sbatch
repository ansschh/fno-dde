#!/bin/bash
#SBATCH --job-name=dde-ddp-{family}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load cuda 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

torchrun --nproc_per_node={n_gpus} --standalone \
    src/train/train_fno_sharded.py \
    --config={config_path} \
    --data_dir={data_dir} \
    --output_dir={output_dir} \
    --device=cuda \
    --seed={seed}

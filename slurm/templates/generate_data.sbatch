#!/bin/bash
#SBATCH --job-name=dde-gen-{family}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time={walltime}
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

module load python/3.10

cd $SLURM_SUBMIT_DIR
python src/datasets/generate_python.py {family} \
    --n_train={n_train} --n_val={n_val} --n_test={n_test} \
    --output_dir={data_dir}/{family} --seed={seed}

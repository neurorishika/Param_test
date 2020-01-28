#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=05:00:00
#SBATCH --job-name=tensorflow
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

cd $SLURM_SUBMIT_DIR

module load python/3.7
source activate tf
python experiment_sweep.py


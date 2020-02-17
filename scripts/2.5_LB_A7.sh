#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=02:00:00
#SBATCH --job-name=tensorflow
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard

cd $SLURM_SUBMIT_DIR

file="/home/collins/Param_test/label"
name=$(cat "$file")
name="${name}_2.5"

module load python/3.7
python /home/collins/Param_test/initExperiment.py '/home/collins/Param_test/Od/OdorA_High.odor' '/home/collins/Param_test/Op/Dur_6000_OdorDur_1000.protocol' '/home/collins/Param_test/Lc/Locust_B.locust' '7' $name '2.5'


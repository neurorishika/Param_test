formatstring = """#!/bin/sh

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

module load python/3.7
python /home/collins/Param_test/initExperiment.py '/home/collins/Param_test/Od/{}.odor' '/home/collins/Param_test/Op/Dur_6000_OdorDur_1000.protocol' '/home/collins/Param_test/Lc/{}.locust' '{}' $name
"""

n_rep_per_odor = 10
odors = ["OdorA_High","OdorB_High","OdorC_High"]
locusts = ["Locust_A","Locust_B","Locust_C","Locust_D","Locust_E"]
for i in range(1,n_rep_per_odor+1):
    for j in odors:
        for k in locusts:
            with open("L{}_{}{}.sh".format(k[-1],j.split('_')[0][-1],i),'w') as file:
                print(formatstring.format(j,k,i),file=file)

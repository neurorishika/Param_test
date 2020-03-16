import numpy as np

formatstring = """#!/bin/sh

#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --job-name=tensorflow
#SBATCH --error=out/job.%J.err
#SBATCH --output=out/job.%J.out
#SBATCH --partition=standard
#SBATCH -a 0-9

cd $SLURM_SUBMIT_DIR

file="/home/collins/Param_test/label"
name=$(cat "$file")
name="${{name}}_{param}"

module load python/3.7
python /home/collins/Param_test/initExperiment.py '/home/collins/Param_test/Od/{od}' '/home/collins/Param_test/Op/Dur_6000_OdorDur_1000.protocol' '/home/collins/Param_test/Lc/{lc}' $SLURM_ARRAY_TASK_ID $name '{param}'
"""

sweep_param = np.linspace(0.0,3.00,13)
for n in sweep_param:
    for od in ["OdorA_High.odor","OdorB_High.odor","OdorC_High.odor"]:
        for lc in ["Locust_A.locust","Locust_B.locust","Locust_C.locust"]:
            with open("lnpn_{}_od_{}_lc_{}.sh".format(np.round(n,2),od.split('_')[0][-1],lc.split('_')[1][0]),'w') as file:
                print(formatstring.format(param=n,od=od,lc=lc),file=file)

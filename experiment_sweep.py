from subprocess import call
import datetime

dt = datetime.datetime.now()

locusts = ['/home/collins/Param_test/Locust_A.locust']#,'/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Locusts/2020/Locust_B.locust']
protocols = ['/home/collins/Param_test/Dur_6000_OdorDur_1000.protocol']#,'/home/iiser/Collins-Saptarshi 2019b/DAMN/A. Odor Protocols/2020/Dur_6000_OdorDur_2000.protocol']
odors = ["/home/collins/Param_test/OdorA_High.odor"]
n_trials = 1	

for l in range(n_trials):
        for j in protocols:
            for k in locusts:
                for i in odors:
                    print(i,j,k,l)
                    call(['python','initExperiment.py',i,j,k,str(l),dt.strftime("%b_%d_T_%H:%M/")])

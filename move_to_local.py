from subprocess import call
import numpy as np
import os

for a in os.listdir('/home/collins/Simulation_Data/'):
	for b in os.listdir('/home/collins/Simulation_Data/'+a+'/'):
		n_splits = len(list(filter(lambda v: 'batch' in v, os.listdir('/home/collins/Simulation_Data/'+a+'/'+b+'/'))))
		n_batch = 1
		overall_state = []
		# Iterate over the generated output files
		for n,i in enumerate(['/home/collins/Simulation_Data/'+a+'/'+b+'/'+"batch"+str(x+1) for x in range(n_splits)]):
			for m,j in enumerate(["_part_"+str(x+1)+".npy" for x in range(n_batch)]):
			# Since the first element in the series was the last output, we remove them
				if n>0 and m>0:
					overall_state.append(np.load(i+j)[1:,:120])
				else:
					overall_state.append(np.load(i+j)[:,:120])
				print(i+j)
		# Concatenate all the matrix to get a single state matrix
		overall_state = np.concatenate(overall_state)
		folder = '/home/collins/clean_data/'+a+'/'+b+'/'
		if not os.path.exists(folder):
			os.makedirs(folder)
		np.save(folder+"AL_output",overall_state[::100,:])

call(['scp','-r',"/home/collins/clean_data","iiser@192.168.8.104:/home/iiser/Collins-Saptarshi 2019b/Param_data"])
	
